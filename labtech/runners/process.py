import logging
import multiprocessing
import signal
import sys
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import QueueHandler
from queue import Empty, Queue
from typing import Callable, Iterator, Optional, Sequence, cast
from uuid import UUID, uuid4

import psutil

from labtech.exceptions import RunnerError
from labtech.tasks import get_direct_dependencies
from labtech.types import LabContext, ResultMeta, ResultsMap, Runner, RunnerBackend, Storage, Task, TaskMonitorInfo, TaskResult
from labtech.utils import LoggerFileProxy, logger

from .base import run_or_load_task


class SerialFuture(Future):

    def __init__(self, fn: Callable, /, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.set_running_or_notify_cancel():
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as ex:
            self.set_exception(ex)
        else:
            self.set_result(result)


class SerialExecutor(Executor):

    def __init__(self) -> None:
        self.futures: list[SerialFuture] = []

    def submit(self, fn: Callable, /, *args, **kwargs):
        future = SerialFuture(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        if cancel_futures:
            for future in self.futures:
                future.cancel()


def wait_for_first_future(futures: Sequence[Future], *, timeout: Optional[float]):
    # If there are any serial futures, ensure at least one is completed.
    serial_futures = [future for future in futures if isinstance(future, SerialFuture)]
    if serial_futures and not serial_futures[0].done():
        serial_futures[0].run()
    return wait(futures, return_when=FIRST_COMPLETED, timeout=timeout)


def init_task_subprocess(log_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Sub-processes should log onto the queue in order to printed
    # in serial by the main process.
    logger.handlers = []
    logger.addHandler(QueueHandler(log_queue))
    sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')
    sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')


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
    """TODO"""

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        self.process_event_queue = multiprocessing.Manager().Queue(-1)
        self.process_monitor = ProcessMonitor(process_event_queue = self.process_event_queue)

        self.log_queue = multiprocessing.Manager().Queue(-1)

        self.serial_executor = SerialExecutor()
        self.executor: Executor
        if max_workers is not None and max_workers == 1:
            self.executor = self.serial_executor
        else:
            self.executor = ProcessPoolExecutor(
                mp_context=self._get_mp_context(),
                max_workers=max_workers,
                initializer=init_task_subprocess,
                initargs=(self.log_queue,),
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
                         storage: Storage, process_event_queue: Queue) -> TaskResult:
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
                task_name=task_name,
                use_cache=use_cache,
                context=context,
                storage=storage
            )
        finally:
            current_process.name = orig_process_name
            process_event_queue.put(ProcessEndEvent(
                task_name=task_name,
            ))

    def start_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        # Always use a serial executor to run tasks on the
        # main process if max_parallel=1
        executor = (
            self.serial_executor
            if task._lt.max_parallel is not None and task._lt.max_parallel == 1
            else self.executor
        )
        future = self._submit_task(
            executor=executor,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            process_event_queue=self.process_event_queue,
        )
        self.future_to_task[future] = task

    def pending_task_count(self) -> int:
        return len(self.future_to_task)

    def wait(self, *, timeout: Optional[float]) -> Iterator[tuple[Task, ResultMeta | Exception]]:
        done, _ = wait_for_first_future(list(self.future_to_task.keys()), timeout=timeout)



        for future in done:
            task = self.future_to_task[future]
            try:
                task_result = future.result()
            except Exception as ex:
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
        for future in self.future_to_task:
            future.cancel()

    def terminate(self) -> None:
        for process in self.executor._processes.values():  # type: ignore[attr-defined]
            process.terminate()
        self.close(wait=False)

    def close(self, *, wait: bool) -> None:
        if self.closed:
            return
        self.serial_executor.shutdown(wait=wait)
        self.executor.shutdown(wait=wait)

    def get_result(self, task: Task) -> TaskResult:
        return self.results_map[task]

    def remove_result(self, task: Task) -> None:
        if task not in self.results_map:
            return
        logger.debug(f"Removing result from in-memory cache for task: '{task}'")
        del self.results_map[task]

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        return self.process_monitor.get_process_infos()

    @abstractmethod
    def _get_mp_context(self) -> multiprocessing.context.BaseContext:
        """TODO"""

    @abstractmethod
    def _submit_task(self, executor: Executor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue) -> Future:
        """TODO: Should call _subprocess_func()"""
        pass


class SpawnProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.context = context
        self.storage = storage

    def _get_mp_context(self) -> multiprocessing.context.SpawnContext:
        return multiprocessing.get_context('spawn')

    def _submit_task(self, executor: Executor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue) -> Future:
        return executor.submit(
            self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map={
                dependency_task: self.results_map[dependency_task]
                for dependency_task in get_direct_dependencies(task)
            },
            # TODO: Allow a task to specify which context keys it needs
            context=self.context,
            storage=self.storage,
            process_event_queue=process_event_queue,
        )


class SpawnRunnerBackend(RunnerBackend):
    """TODO"""

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


_RUNNER_FORK_MEMORY: dict[UUID, RunnerMemory] = {}


class ForkProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.uuid = uuid4()
        _RUNNER_FORK_MEMORY[self.uuid] = RunnerMemory(
            context=context,
            storage=storage,
        )

    def _get_mp_context(self) -> multiprocessing.context.ForkContext:
        return multiprocessing.get_context('fork')

    @staticmethod
    def _fork_subprocess_func(*, _subprocess_func: Callable, task: Task, task_name: str,
                              use_cache: bool, process_event_queue: Queue, uuid: UUID,
                              results_map: ResultsMap) -> TaskResult:
        runner_memory = _RUNNER_FORK_MEMORY[uuid]
        return _subprocess_func(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            context=runner_memory.context,
            storage=runner_memory.storage,
            results_map=results_map,
            process_event_queue=process_event_queue,
        )

    def _submit_task(self, executor: Executor, task: Task, task_name: str,
                     use_cache: bool, process_event_queue: Queue) -> Future:
        return executor.submit(
            self._fork_subprocess_func,
            _subprocess_func=self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            process_event_queue=process_event_queue,
            uuid=self.uuid,
            # TODO: Ideally we should be able to share the results_map
            # via _RUNNER_FORK_MEMORY, but concurrent.futures forks
            # all processes up front instead of as each task is
            # started, so we'd need to move away from
            # concurrent.futures to fork for each task. While
            # concurrent.futures can't fork on-demand because it uses
            # a manager thread, we should be able to safely do it if
            # we don't use any threads. See:
            # https://github.com/python/cpython/issues/90622#issuecomment-1093942931
            results_map={
                dependency_task: self.results_map[dependency_task]
                for dependency_task in get_direct_dependencies(task)
            },
        )

    def close(self, *, wait: bool) -> None:
        del _RUNNER_FORK_MEMORY[self.uuid]
        super().close(wait=wait)


class ForkRunnerBackend(RunnerBackend):
    """TODO"""

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> ForkProcessRunner:
        return ForkProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )
