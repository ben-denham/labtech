import logging
import multiprocessing
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from logging.handlers import QueueHandler
from queue import Empty, Queue
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, cast
from uuid import UUID, uuid4

import psutil

from labtech.exceptions import RunnerError, TaskDiedError
from labtech.tasks import get_direct_dependencies
from labtech.types import LabContext, ResultMeta, ResultsMap, Runner, RunnerBackend, Storage, Task, TaskMonitorInfo, TaskResult
from labtech.utils import LoggerFileProxy, logger

from .base import run_or_load_task


class FutureStateError(Exception):
    pass


class FutureState(StrEnum):
    PENDING = auto()
    CANCELLED = auto()
    FINISHED = auto()


@dataclass(eq=False)
class Future:
    """Representation of a result to be returned in the future by runner.

    A Future's state transitions between states according to the following
    finite state machine:

    * A Future starts in a PENDING state
    * A PENDING Future can be transitioned to FINISHED by calling
      set_result() or set_exception()
    * Any Future can be transitioned to CANCELLED by calling cancel()
    * result() can only be called on a FINISHED Future, and it will either
      return the result set by set_result() or raise the exception set by
      set_exception()

    """
    state: FutureState = FutureState.PENDING
    _ex: Optional[BaseException] = None
    _result: Optional[Any] = None

    @property
    def done(self) -> bool:
        return self.state in {FutureState.FINISHED, FutureState.CANCELLED}

    def set_result(self, result: Any):
        if self.done:
            raise FutureStateError(f'Attempted to set a result on a {self.state} future.')
        self._result = result
        self.state = FutureState.FINISHED

    def set_exception(self, ex: BaseException):
        if self.done:
            raise FutureStateError(f'Attempted to set an exception on a {self.state} future.')
        self._ex = ex
        self.state = FutureState.FINISHED

    def cancel(self):
        self.state = FutureState.CANCELLED

    def result(self) -> Any:
        if self.state != FutureState.FINISHED:
            raise FutureStateError(f'Attempted to get result from a {self.state} future.')
        if self._ex is not None:
            raise self._ex
        return self._result


def split_done_futures(futures: Sequence[Future]) -> tuple[list[Future], list[Future]]:
    done_futures = []
    not_done_futures = []
    for future in futures:
        if future.done:
            done_futures.append(future)
        else:
            not_done_futures.append(future)
    return (done_futures, not_done_futures)


class Executor(ABC):

    @abstractmethod
    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        """Schedule the given fn to be called with the given *args and
        **kwargs, and return a Future that will be updated with the
        outcome of function call."""

    def stop(self, *, wait: bool):
        """If wait=True, cancel all PENDING futures and wait() until
        all running futures are FINISHED. If wait=False, then also
        cancel all RUNNING futures and immediately terminate their
        execution."""

    @abstractmethod
    def wait(self, futures: Sequence[Future], *, timeout_seconds: Optional[float]) -> tuple[list[Future], list[Future]]:
        """Wait up to timeout_seconds or until at least one of the
        given futures is done, then return a list of futures in a done
        state and a list of futures in all other states."""


@dataclass(frozen=True)
class Thunk:
    fn: Callable
    args: Sequence[Any]
    kwargs: Mapping[str, Any]


def _subprocess_target(*, future: Future, thunk: Thunk, result_queue: Queue) -> None:
    try:
        result = thunk.fn(*thunk.args, **thunk.kwargs)
    except BaseException as ex:
        result_queue.put((future, ex))
    else:
        result_queue.put((future, result))


class ProcessExecutor(Executor):

    def __init__(self, mp_context: multiprocessing.context.BaseContext, max_workers: Optional[int]):
        self.mp_context = mp_context
        self.max_workers = max_workers
        self._pending_future_to_thunk: dict[Future, Thunk] = {}
        self._running_future_to_process: dict[Future, multiprocessing.Process] = {}
        # Use a Manager().Queue() to be able to share with subprocesses
        self._result_queue: Queue = multiprocessing.Manager().Queue(-1)

    def _start_processes(self):
        """Start processes for the oldest pending futures to bring
        running process count up to max_workers."""
        if self.max_workers is None:
            start_count = len(self._pending_future_to_thunk)
        else:
            start_count = max(0, self.max_workers - len(self._future_to_process))
        futures_to_start = list(self._pending_future_to_thunk.keys())[:start_count]
        for future in futures_to_start:
            thunk = self._pending_future_to_thunk[future]
            del self._pending_future_to_thunk[future]
            process = multiprocessing.Process(
                target=_subprocess_target,
                kwargs=dict(
                    future=future,
                    thunk=thunk,
                    result_queue=self._result_queue,
                ),
            )
            self._running_future_to_process[future] = process
            process.start()

    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        future = Future()
        self._pending_future_to_thunk[future] = Thunk(fn=fn, args=args, kwargs=kwargs)
        self._start_processes()
        return future

    def stop(self, *, wait: bool):
        # Cancel pending futures
        pending_futures = list(self._pending_future_to_thunk.keys())
        for future in pending_futures:
            future.cancel()
            del self._pending_future_to_thunk[future]

        # Wait for or terminate+cancel running futures
        future_process_pairs = list(self._running_future_to_process.items())
        for future, process in future_process_pairs:
            if wait:
                process.join()
            else:
                process.terminate()
                future.cancel()
                del self._running_future_to_process[future]

    def _consume_result_queue(self, *, timeout_seconds: Optional[float]):
        # Avoid race condition of a process finishing after we have
        # consumed the result_queue by fetching process statuses
        # before checking for process completion.
        dead_process_futures = [
            future for future, process in self._running_future_to_process.items()
            if not process.is_alive()
        ]

        wait_timeout = True
        while True:
            try:
                future, result_or_ex = self._result_queue.get(wait_timeout, timeout=timeout_seconds)
            except Empty:
                break

            # Don't wait for the timeout on subsequent calls to
            # self._result_queue.get()
            wait_timeout = False

            if not future.done:
                del self._running_future_to_process[future]
                if isinstance(result_or_ex, BaseException):
                    future.set_exception(result_or_ex)
                else:
                    future.set_result(result_or_ex)

        # If any processes have died without the future being
        # cancelled or finished, then set an exception for it.
        for future in dead_process_futures:
            if future.done:
                continue
            future.set_exception(TaskDiedError())
            del self._running_future_to_process[future]

    def wait(self, futures: Sequence[Future], *, timeout_seconds: Optional[float]) -> tuple[list[Future], list[Future]]:
        self._consume_result_queue(timeout_seconds=timeout_seconds)
        return split_done_futures(futures)


class SerialExecutor(Executor):

    def __init__(self) -> None:
        self._pending_future_to_thunk: dict[Future, Thunk] = {}

    def submit(self, fn: Callable, /, *args, **kwargs):
        future = Future()
        self._pending_future_to_thunk[future] = Thunk(fn=fn, args=args, kwargs=kwargs)
        return future

    def stop(self, *, wait: bool):
        # Cancel pending futures
        pending_futures = list(self._pending_future_to_thunk.keys())
        for future in pending_futures:
            future.cancel()
            del self._pending_future_to_thunk[future]

    def _run_future(self, future: Future):
        thunk = self._pending_future_to_thunk[future]
        try:
            result = thunk.fn(*thunk.args, **thunk.kwargs)
        except BaseException as ex:
            future.set_exception(ex)
        else:
            future.set_result(result)
        del self._pending_future_to_thunk[future]

    def wait(self, futures: Sequence[Future], *, timeout_seconds: Optional[float]) -> tuple[list[Future], list[Future]]:
        if not futures:
            return [], []

        # Ensure at least one future is completed.
        completed_futures = [future for future in futures if future.done]
        if not completed_futures:
            self._run_future(futures[0])

        return split_done_futures(futures)


class ProcessEvent:
    pass


@dataclass(frozen=True)
class ProcessStartEvent(ProcessEvent):
    task_name: str
    pid: int
    use_cache: bool


@dataclass(frozen=True)
class ProcessEndEvent(ProcessEvent):
    task_name: str


class ProcessMonitor:

    def __init__(self, *, process_event_queue: Queue):
        self.process_event_queue = process_event_queue
        self.active_process_events: dict[str, ProcessStartEvent] = {}
        self.active_processes: dict[str, psutil.Process] = {}

    def _consume_monitor_queue(self):
        while True:
            try:
                event = self.process_event_queue.get_nowait()
            except Empty:
                break

            if isinstance(event, ProcessStartEvent):
                self.active_process_events[event.task_name] = event
            elif isinstance(event, ProcessEndEvent):
                if event.task_name in self.active_process_events:
                    del self.active_process_events[event.task_name]
                    if event.task_name in self.active_processes:
                        del self.active_processes[event.task_name]
            else:
                raise RunnerError(f'Unexpected process event: {event}')

    def _get_process_info(self, start_event: ProcessStartEvent) -> Optional[TaskMonitorInfo]:
        pid = start_event.pid
        try:
            if start_event.task_name not in self.active_processes:
                self.active_processes[start_event.task_name] = psutil.Process(pid)
            process = self.active_processes[start_event.task_name]
            with process.oneshot():
                start_datetime = datetime.fromtimestamp(process.create_time())
                threads = process.num_threads()
                cpu_percent = process.cpu_percent()
                memory_rss_percent = process.memory_percent('rss')
                memory_vms_percent = process.memory_percent('vms')
                children = process.children(recursive=True)
            for child in children:
                with child.oneshot():
                    threads += child.num_threads()
                    cpu_percent += child.cpu_percent()
                    memory_rss_percent += child.memory_percent('rss')
                    memory_vms_percent += child.memory_percent('vms')
        except psutil.NoSuchProcess:
            return None
        return {
            'name': start_event.task_name,
            'pid': pid,
            'status': ('loading' if start_event.use_cache else 'running'),
            'start_time': (start_datetime, start_datetime.strftime('%H:%M:%S')),
            'children': len(children),
            'threads': threads,
            'cpu': (cpu_percent, f'{cpu_percent/100:.1%}'),
            'rss': (memory_rss_percent, f'{memory_rss_percent/100:.1%}'),
            'vms': (memory_vms_percent, f'{memory_vms_percent/100:.1%}'),
        }

    def get_process_infos(self) -> list[TaskMonitorInfo]:
        self._consume_monitor_queue()
        process_infos: list[TaskMonitorInfo] = []
        for start_event in self.active_process_events.values():
            process_info = self._get_process_info(start_event)
            if process_info is not None:
                process_infos.append(process_info)
        return process_infos


class ProcessRunner(Runner, ABC):
    """Base class for Runner's based on Python multiprocessing."""

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        self.process_event_queue = multiprocessing.Manager().Queue(-1)
        self.process_monitor = ProcessMonitor(process_event_queue = self.process_event_queue)

        self.log_queue = multiprocessing.Manager().Queue(-1)

        self.executor: Executor
        if max_workers is not None and max_workers == 1:
            self.executor = SerialExecutor()
        else:
            self.executor = ProcessExecutor(
                mp_context=self._get_mp_context(),
                max_workers=max_workers,
            )

        self.results_map: dict[Task, TaskResult] = {}
        self.future_to_task: dict[Future, Task] = {}
        self.closed = False

    def _consume_log_queue(self):
        # See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        while True:
            try:
                record = self.log_queue.get_nowait()
            except Empty:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    @staticmethod
    def _subprocess_func(*, task: Task, task_name: str, use_cache: bool,
                         results_map: ResultsMap, context: LabContext,
                         storage: Storage, process_event_queue: Queue,
                         log_queue: Queue) -> TaskResult:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Sub-processes should log onto the queue in order to printed
        # in serial by the main process.
        logger.handlers = []
        logger.addHandler(QueueHandler(log_queue))
        # Ignore type errors for type of value used to override stdout and stderr
        sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')  # type: ignore[assignment]
        sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')  # type: ignore[assignment]

        orig_process_name = multiprocessing.current_process().name
        try:
            current_process = multiprocessing.current_process()
            current_process.name = task_name

            process_event_queue.put(ProcessStartEvent(
                task_name=task_name,
                pid=cast(int, current_process.pid),
                use_cache=use_cache,
            ))

            for dependency_task in get_direct_dependencies(task):
                dependency_task._set_results_map(results_map)

            return run_or_load_task(
                task=task,
                use_cache=use_cache,
                context=context,
                storage=storage
            )
        finally:
            current_process.name = orig_process_name
            process_event_queue.put(ProcessEndEvent(
                task_name=task_name,
            ))

    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        future = self._submit_task(
            executor=self.executor,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            process_event_queue=self.process_event_queue,
            log_queue=self.log_queue,
        )
        self.future_to_task[future] = task

    def wait(self, *, timeout_seconds: Optional[float]) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        done, _ = self.executor.wait(list(self.future_to_task.keys()), timeout_seconds=timeout_seconds)
        for future in done:
            task = self.future_to_task[future]
            try:
                task_result = future.result()
            except BaseException as ex:
                yield (task, ex)
            else:
                self.results_map[task] = task_result
                yield (task, task_result.meta)
        self.future_to_task = {
            future: self.future_to_task[future]
            for future in self.future_to_task
            if future not in done
        }

    def close(self, *, wait: bool) -> None:
        if self.closed:
            return
        self.executor.stop(wait=wait)
        if wait:
            # If we wait for tasks to finish, then handle their
            # results.
            self.wait(timeout_seconds=0)
        self.closed = True

    def submitted_task_count(self) -> int:
        return len(self.future_to_task)

    def get_result(self, task: Task) -> TaskResult:
        return self.results_map[task]

    def remove_results(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            if task not in self.results_map:
                return
            logger.debug(f"Removing result from in-memory cache for task: '{task}'")
            del self.results_map[task]

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        return self.process_monitor.get_process_infos()

    @abstractmethod
    def _get_mp_context(self) -> multiprocessing.context.BaseContext:
        """Return a multiprocessing context from which to start subprocesses."""

    @abstractmethod
    def _submit_task(self, executor: Executor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> Future:
        """Should submit the execution of self._subprocess_func() on the given
        task to the given executor and return the resulting Future.

        Sub-classes can use the implementation of this method to load
        or otherwise prepare context or dependency results for the task.

        """


class SpawnProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.context = context
        self.storage = storage

    def _get_mp_context(self) -> multiprocessing.context.SpawnContext:
        return multiprocessing.get_context('spawn')

    def _submit_task(self, executor: Executor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> Future:
        context: LabContext = {}
        results_map: dict[Task, TaskResult] = {}
        if not use_cache:
            # Only transfer context and results to the subprocess if
            # we are going to run the task (and not just load its
            # result from cache). And allow the task to filter the
            # context to only what it needs.
            context = task.filter_context(self.context)
            results_map = {
                dependency_task: self.results_map[dependency_task]
                for dependency_task in get_direct_dependencies(task)
            }
        return executor.submit(
            self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map=results_map,
            context=context,
            storage=self.storage,
            process_event_queue=process_event_queue,
            log_queue=log_queue,
        )


class SpawnRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs each task in a spawned subprocess.

    The context and dependency task results are copied/duplicated into
    the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> SpawnProcessRunner:
        return SpawnProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )


@dataclass
class RunnerMemory:
    context: LabContext
    storage: Storage
    results_map: ResultsMap


_RUNNER_FORK_MEMORY: dict[UUID, RunnerMemory] = {}


class ForkProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.uuid = uuid4()
        _RUNNER_FORK_MEMORY[self.uuid] = RunnerMemory(
            context=context,
            storage=storage,
            results_map=self.results_map,
        )

    def _get_mp_context(self) -> multiprocessing.context.ForkContext:
        return multiprocessing.get_context('fork')

    @staticmethod
    def _fork_subprocess_func(*, _subprocess_func: Callable, task: Task, task_name: str,
                              use_cache: bool, process_event_queue: Queue, log_queue: Queue,
                              uuid: UUID) -> TaskResult:
        runner_memory = _RUNNER_FORK_MEMORY[uuid]
        return _subprocess_func(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            context=task.filter_context(runner_memory.context),
            storage=runner_memory.storage,
            results_map=runner_memory.results_map,
            process_event_queue=process_event_queue,
            log_queue=log_queue,
        )

    def _submit_task(self, executor: Executor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> Future:
        return executor.submit(
            self._fork_subprocess_func,
            _subprocess_func=self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            process_event_queue=process_event_queue,
            log_queue=log_queue,
            uuid=self.uuid,
        )

    def close(self, *, wait: bool) -> None:
        del _RUNNER_FORK_MEMORY[self.uuid]
        super().close(wait=wait)


class ForkRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs each task in a forked subprocess.

    The context and dependency task results are copied/duplicated into
    the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> ForkProcessRunner:
        return ForkProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )
