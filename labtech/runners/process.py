from __future__ import annotations

import functools
import logging
import multiprocessing
import os
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, auto
from itertools import count
from logging.handlers import QueueHandler
from queue import Empty
from threading import Thread
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import psutil

from labtech.exceptions import RunnerError, TaskDiedError
from labtech.monitor import get_process_info
from labtech.tasks import get_direct_dependencies
from labtech.types import Runner, RunnerBackend
from labtech.utils import LoggerFileProxy, get_supported_start_methods, logger

from .base import run_or_load_task

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from multiprocessing.context import BaseContext, SpawnContext
    from queue import Queue
    from typing import Any
    from uuid import UUID

    from labtech.types import LabContext, ResultMeta, ResultsMap, Storage, Task, TaskMonitorInfo, TaskResult

    if sys.platform != 'win32':
        from multiprocessing.context import ForkContext
    else:
        ForkContext = BaseContext


class FutureStateError(Exception):
    pass


class FutureState(StrEnum):
    PENDING = auto()
    CANCELLED = auto()
    FINISHED = auto()


@dataclass
class Future:
    """Representation of a result to be returned in the future by a runner.

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
    # Auto-incrementing ID (does not need to be process-safe because
    # all futures are generated in the main process):
    id: int = field(default_factory=count().__next__, init=False)
    _state: FutureState = FutureState.PENDING
    _ex: BaseException | None = None
    _result: Any | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return id(self.id) == id(other.id)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def done(self) -> bool:
        return self._state in {FutureState.FINISHED, FutureState.CANCELLED}

    @property
    def cancelled(self) -> bool:
        return self._state == FutureState.CANCELLED

    def set_result(self, result: Any):
        if self.done:
            raise FutureStateError(f'Attempted to set a result on a {self._state} future.')
        self._result = result
        self._state = FutureState.FINISHED

    def set_exception(self, ex: BaseException):
        if self.done:
            raise FutureStateError(f'Attempted to set an exception on a {self._state} future.')
        self._ex = ex
        self._state = FutureState.FINISHED

    def cancel(self):
        self._state = FutureState.CANCELLED

    def result(self) -> Any:
        if self._state != FutureState.FINISHED:
            raise FutureStateError(f'Attempted to get result from a {self._state} future.')
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


def _subprocess_target(*, future_id: int, thunk: Callable[[], Any], result_queue: Queue) -> None:
    try:
        result = thunk()
    except BaseException as ex:
        result_queue.put((future_id, ex))
    else:
        result_queue.put((future_id, result))


class ProcessExecutor:

    def __init__(self, mp_context: BaseContext, max_workers: int | None):
        self.mp_context = mp_context
        self.max_workers = (os.cpu_count() or 1) if max_workers is None else max_workers
        self._pending_future_to_thunk: dict[Future, Callable[[], Any]] = {}
        self._running_id_to_future_and_process: dict[int, tuple[Future, multiprocessing.Process]] = {}
        # Use a Manager().Queue() to be able to share with subprocesses
        self._result_queue: Queue = multiprocessing.Manager().Queue(-1)

    def _start_processes(self):
        """Start processes for the oldest pending futures to bring
        running process count up to max_workers."""
        start_count = max(0, self.max_workers - len(self._running_id_to_future_and_process))
        futures_to_start = list(self._pending_future_to_thunk.keys())[:start_count]
        for future in futures_to_start:
            thunk = self._pending_future_to_thunk[future]
            del self._pending_future_to_thunk[future]
            process = multiprocessing.Process(
                target=_subprocess_target,
                kwargs=dict(
                    future_id=future.id,
                    thunk=thunk,
                    result_queue=self._result_queue,
                ),
            )
            self._running_id_to_future_and_process[future.id] = (future, process)
            process.start()

    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        """Schedule the given fn to be called with the given *args and
        **kwargs, and return a Future that will be updated with the
        outcome of function call."""
        future = Future()
        self._pending_future_to_thunk[future] = functools.partial(fn, *args, **kwargs)
        self._start_processes()
        return future

    def cancel(self) -> None:
        """Cancel all pending futures."""
        pending_futures = list(self._pending_future_to_thunk.keys())
        for future in pending_futures:
            future.cancel()
            del self._pending_future_to_thunk[future]

    def stop(self) -> None:
        """Cancel all running futures and immediately terminate their execution."""
        future_process_pairs = list(self._running_id_to_future_and_process.values())
        for future, process in future_process_pairs:
            process.terminate()
            future.cancel()
            del self._running_id_to_future_and_process[future.id]

    def _consume_result_queue(self, *, timeout_seconds: float | None):
        # Avoid race condition of a process finishing after we have
        # consumed the result_queue by fetching process statuses
        # before checking for process completion.
        dead_process_futures = [
            future for future, process in self._running_id_to_future_and_process.values()
            if not process.is_alive()
        ]

        def _consume():
            inner_timeout_seconds = timeout_seconds
            while True:
                try:
                    future_id, result_or_ex = self._result_queue.get(True, timeout=inner_timeout_seconds)
                except Empty:
                    break

                # Don't wait for the timeout on subsequent calls to
                # self._result_queue.get()
                inner_timeout_seconds = 0

                future, _ = self._running_id_to_future_and_process[future_id]
                del self._running_id_to_future_and_process[future_id]
                if not future.done:
                    if isinstance(result_or_ex, BaseException):
                        future.set_exception(result_or_ex)
                    else:
                        future.set_result(result_or_ex)

        # Consume the result queue in a thread so that it is not
        # interrupt by KeyboardInterrupt, which can result in us not
        # fully processing a completed result. Despite the fact we are
        # using subprocesses, starting a thread at this point should
        # be safe because we will not start any subprocesses while
        # this is running?
        consumer_thread = Thread(target=_consume)
        consumer_thread.start()
        consumer_thread.join()

        # If any processes have died without the future being
        # cancelled or finished, then set an exception for it.
        for future in dead_process_futures:
            if future.done:
                continue
            future.set_exception(TaskDiedError())
            del self._running_id_to_future_and_process[future.id]

    def wait(self, futures: Sequence[Future], *, timeout_seconds: float | None) -> tuple[list[Future], list[Future]]:
        """Wait up to timeout_seconds or until at least one of the
        given futures is done, then return a list of futures in a done
        state and a list of futures in all other states."""
        self._consume_result_queue(timeout_seconds=timeout_seconds)
        # Having consumed completed results, start new processes
        self._start_processes()
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
        self.active_processes_and_children: dict[str, tuple[psutil.Process, dict[int, psutil.Process]]] = {}

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
                    if event.task_name in self.active_processes_and_children:
                        del self.active_processes_and_children[event.task_name]
            else:
                raise RunnerError(f'Unexpected process event: {event}')

    def _get_process_info(self, start_event: ProcessStartEvent) -> TaskMonitorInfo | None:
        pid = start_event.pid
        try:
            if start_event.task_name not in self.active_processes_and_children:
                self.active_processes_and_children[start_event.task_name] = (psutil.Process(pid), {})
            process, previous_child_processes = self.active_processes_and_children[start_event.task_name]
        except psutil.NoSuchProcess:
            return None

        info, child_processes = get_process_info(
            process,
            previous_child_processes=previous_child_processes,
            name=start_event.task_name,
            status=('loading' if start_event.use_cache else 'running'),
        )
        self.active_processes_and_children[start_event.task_name] = (process, child_processes)
        return info

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

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None):
        self.process_event_queue = multiprocessing.Manager().Queue(-1)
        self.process_monitor = ProcessMonitor(process_event_queue = self.process_event_queue)
        self.log_queue = multiprocessing.Manager().Queue(-1)
        self.executor = ProcessExecutor(
            mp_context=self._get_mp_context(),
            max_workers=max_workers,
        )

        self.results_map: dict[Task, TaskResult] = {}
        self.future_to_task: dict[Future, Task] = {}

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
                         results_map: ResultsMap, filtered_context: LabContext,
                         storage: Storage, process_event_queue: Queue,
                         log_queue: Queue) -> TaskResult:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Subprocesses should log onto the queue in order to printed
        # in serial by the main process.
        logger.handlers = []
        logger.addHandler(QueueHandler(log_queue))
        # Ignore type errors for type of value used to override stdout and stderr
        sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')  # type: ignore[assignment]
        sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')  # type: ignore[assignment]

        try:
            current_process = multiprocessing.current_process()
            process_event_queue.put(ProcessStartEvent(
                task_name=task_name,
                pid=cast('int', current_process.pid),
                use_cache=use_cache,
            ))

            for dependency_task in get_direct_dependencies(task, all_identities=True):
                dependency_task._set_results_map(results_map)

            orig_process_name = current_process.name
            try:
                current_process.name = task_name
                return run_or_load_task(
                    task=task,
                    use_cache=use_cache,
                    filtered_context=filtered_context,
                    storage=storage
                )
            finally:
                current_process.name = orig_process_name
        finally:
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

    def wait(self, *, timeout_seconds: float | None) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        self._consume_log_queue()
        done, _ = self.executor.wait(list(self.future_to_task.keys()), timeout_seconds=timeout_seconds)
        for future in done:
            task = self.future_to_task[future]
            if future.cancelled:
                continue
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

    def cancel(self) -> None:
        self.executor.cancel()

    def stop(self) -> None:
        self.executor.stop()

    def close(self) -> None:
        pass

    def pending_task_count(self) -> int:
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
    def _get_mp_context(self) -> BaseContext:
        """Return a multiprocessing context from which to start subprocesses."""

    @abstractmethod
    def _submit_task(self, executor: ProcessExecutor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> Future:
        """Should submit the execution of self._subprocess_func() on the given
        task to the given executor and return the resulting Future.

        Sub-classes can use the implementation of this method to load
        or otherwise prepare context or dependency results for the task.

        """


class SpawnProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.context = context
        self.storage = storage

    def _get_mp_context(self) -> SpawnContext:
        return multiprocessing.get_context('spawn')

    def _submit_task(self, executor: ProcessExecutor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> Future:
        filtered_context: LabContext = {}
        results_map: dict[Task, TaskResult] = {}
        if not use_cache:
            # In order to minimise memory use, only transfer context
            # and results to the subprocess if we are going to run the
            # task (and not just load its result from cache) and allow
            # the task to filter the context to only what it needs.
            filtered_context = task.filter_context(self.context)
            results_map = {
                dependency_task: self.results_map[dependency_task]
                for dependency_task in get_direct_dependencies(task, all_identities=False)
            }
        return executor.submit(
            self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map=results_map,
            filtered_context=filtered_context,
            storage=self.storage,
            process_event_queue=process_event_queue,
            log_queue=log_queue,
        )


class SpawnRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs each task in a spawned subprocess.

    The required context and dependency task results are
    copied/duplicated into the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> SpawnProcessRunner:
        if 'spawn' not in get_supported_start_methods():
            raise RunnerError(
                ("The 'spawn' start method for processes is not supported by your operating system. "
                 "Please specify a system-compatible runner_backend.")
            )

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

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.uuid = uuid4()
        _RUNNER_FORK_MEMORY[self.uuid] = RunnerMemory(
            context=context,
            storage=storage,
            results_map=self.results_map,
        )

    def _get_mp_context(self) -> ForkContext:
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
            filtered_context=task.filter_context(runner_memory.context),
            storage=runner_memory.storage,
            results_map=runner_memory.results_map,
            process_event_queue=process_event_queue,
            log_queue=log_queue,
        )

    def _submit_task(self, executor: ProcessExecutor, task: Task, task_name: str,
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

    def close(self) -> None:
        try:
            del _RUNNER_FORK_MEMORY[self.uuid]
        except KeyError:
            # uuid not may be found if close() is called twice.
            pass


class ForkRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs each task in a forked subprocess.

    The context and dependency task results are shared in-memory
    between each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ForkProcessRunner:
        if 'fork' not in get_supported_start_methods():
            raise RunnerError(
                ("The 'fork' start method for processes is not supported by your operating system. "
                 "Try switching to runner_backend='spawn' or specify another system-compatible runner_backend.")
            )

        return ForkProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )
