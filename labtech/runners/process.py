from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from logging.handlers import QueueHandler
from queue import Empty
from time import monotonic
from typing import TYPE_CHECKING, Generic, TypeVar, cast
from uuid import uuid4

import psutil

from labtech.exceptions import RunnerError
from labtech.monitor import get_process_info
from labtech.tasks import get_direct_dependencies
from labtech.types import Runner, RunnerBackend
from labtech.utils import LoggerFileProxy, get_supported_start_methods, is_interactive, logger

from ._process_executor import ExecutorFuture, ProcessExecutor
from .base import run_or_load_task

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from multiprocessing.context import BaseContext
    from queue import Queue
    from uuid import UUID

    from labtech.types import LabContext, ResultMeta, ResultsMap, Storage, Task, TaskMonitorInfo, TaskResult

Future = ConcurrentFuture | ExecutorFuture
FutureT = TypeVar('FutureT', bound=Future, covariant=True)


@dataclass(frozen=True)
class TaskStartEvent:
    task_name: str
    pid: int
    use_cache: bool


@dataclass(frozen=True)
class TaskEndEvent:
    task_name: str


class ProcessMonitor:

    def __init__(self, *, task_event_queue: Queue):
        self.task_event_queue = task_event_queue
        self.active_task_events: dict[str, TaskStartEvent] = {}
        self.active_processes_and_children: dict[str, tuple[psutil.Process, dict[int, psutil.Process]]] = {}

    def _consume_monitor_queue(self):
        while True:
            try:
                event = self.task_event_queue.get_nowait()
            except Empty:
                break

            if isinstance(event, TaskStartEvent):
                self.active_task_events[event.task_name] = event
            elif isinstance(event, TaskEndEvent):
                if event.task_name in self.active_task_events:
                    del self.active_task_events[event.task_name]
                    if event.task_name in self.active_processes_and_children:
                        del self.active_processes_and_children[event.task_name]
            else:
                raise RunnerError(f'Unexpected task event: {event}')

    def _get_process_info(self, start_event: TaskStartEvent) -> TaskMonitorInfo | None:
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
        for start_event in self.active_task_events.values():
            process_info = self._get_process_info(start_event)
            if process_info is not None:
                process_infos.append(process_info)
        return process_infos


def _task_subprocess_func(*, task: Task, task_name: str, use_cache: bool,
                          results_map: ResultsMap,
                          filtered_context: LabContext, storage: Storage,
                          task_event_queue: Queue, log_queue: Queue) -> TaskResult:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Subprocesses should log onto the queue in order to printed
    # in serial by the main process.
    logger.handlers = []
    logger.addHandler(QueueHandler(log_queue))
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    # Ignore type errors for type of value used to override stdout and stderr
    sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')  # type: ignore[assignment]
    sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')  # type: ignore[assignment]

    try:
        current_process = multiprocessing.current_process()
        task_event_queue.put(TaskStartEvent(
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
        task_event_queue.put(TaskEndEvent(
            task_name=task_name,
        ))
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


class ProcessRunner(Runner, Generic[FutureT], ABC):
    """Runner based on Python multiprocessing."""

    def __init__(self) -> None:
        self.last_consume_log = monotonic()
        self.log_queue = multiprocessing.Manager().Queue(-1)
        self.task_event_queue = multiprocessing.Manager().Queue(-1)
        self.process_monitor = ProcessMonitor(task_event_queue = self.task_event_queue)

        self.results_map: dict[Task, TaskResult] = {}
        self.future_to_task: dict[FutureT, Task] = {}

    def _consume_log_queue(self):
        # See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        while True:
            try:
                record = self.log_queue.get_nowait()
            except Empty:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        future = self._schedule_subprocess(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
        )
        self.future_to_task[future] = task

    def wait(self, *, timeout_seconds: float | None) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        # Consume logs at most every half second.
        if (monotonic() - self.last_consume_log) >= 0.5:
            self._consume_log_queue()
            self.last_consume_log = monotonic()

        done = self._get_completed_futures(
            futures=list(self.future_to_task.keys()),
            timeout_seconds=timeout_seconds,
        )
        for future in done:
            task = self.future_to_task[future]
            if future.cancelled():
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

    def close(self) -> None:
        self._consume_log_queue()
        self._close_executor()

    def pending_task_count(self) -> int:
        return len(self.future_to_task)

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        return self.process_monitor.get_process_infos()

    def get_result(self, task: Task) -> TaskResult:
        return self.results_map[task]

    def remove_results(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            if task not in self.results_map:
                return
            logger.debug(f"Removing result from in-memory cache for task: '{task}'")
            del self.results_map[task]

    @abstractmethod
    def _schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool) -> FutureT:
        """Should submit the execution of _task_subprocess_func()
        for the given task in a subprocess and return the resulting Future.

        The implementation of this method to load or otherwise prepare
        context or dependency results for the task.

        """

    @abstractmethod
    def _get_completed_futures(self, futures: list[FutureT], timeout_seconds: float | None) -> list[FutureT]:
        """Return a sub-sequence of the given futures that have been completed."""

    @abstractmethod
    def cancel(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def _close_executor(self) -> None:
        """Stop all currently running subprocesses."""


def _spawn_start_method_check() -> None:
    if 'spawn' not in get_supported_start_methods():
        raise RunnerError(
            ("The 'spawn' start method for processes is not supported by your operating system. "
             "Please specify a system-compatible runner_backend.")
        )


def _spawn_interactive_main_check(cls: type, task: Task) -> None:
    if is_interactive() and task.__class__.__module__ == '__main__':
        raise RunnerError(
            (f'Unable to submit {task.__class__.__qualname__} tasks to '
             f'{cls.__qualname__} because the task type is defined in the '
             '__main__ module from an interactive Python session. '
             'Please define your task types in a separate `.py` Python '
             'module file. For details, see: '
             'https://ben-denham.github.io/labtech/cookbook/#spawn-interactive-main')
        )


def _fork_start_method_check() -> None:
    if 'fork' not in get_supported_start_methods():
        raise RunnerError(
            ("The 'fork' start method for processes is not supported by your operating system. "
             "Try switching to runner_backend='spawn' or specify another system-compatible runner_backend.")
        )


# === Subprocess Pools  ===

class PoolProcessRunner(ProcessRunner[ConcurrentFuture], ABC):

    def __init__(self, *, mp_context: BaseContext, max_workers: int | None) -> None:
        super().__init__()
        self.concurrent_executor = ProcessPoolExecutor(
            mp_context=mp_context,
            max_workers=max_workers,
        )

    def _get_completed_futures(self, futures: list[ConcurrentFuture], timeout_seconds: float | None) -> list[ConcurrentFuture]:
        done, _ = wait_futures(futures, timeout=timeout_seconds, return_when=FIRST_COMPLETED)
        return list(done)

    def cancel(self) -> None:
        self.concurrent_executor.shutdown(wait=True, cancel_futures=True)

    def stop(self) -> None:
        self.concurrent_executor.shutdown(wait=True, cancel_futures=True)
        for process in self.concurrent_executor._processes.values():
            process.terminate()

    def _close_executor(self) -> None:
        self.concurrent_executor.shutdown(wait=True)


class SpawnPoolProcessRunner(PoolProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> None:
        super().__init__(
            mp_context=multiprocessing.get_context('spawn'),
            max_workers=max_workers,
        )
        self.context = context
        self.storage = storage

    def _schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool) -> ConcurrentFuture:
        _spawn_interactive_main_check(self.__class__, task)

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

        return self.concurrent_executor.submit(
            _task_subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map=results_map,
            filtered_context=filtered_context,
            storage=self.storage,
            task_event_queue=self.task_event_queue,
            log_queue=self.log_queue,
        )


class SpawnPoolRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs tasks on a pool of spawned subprocesses.

    The required context and dependency task results are
    copied/duplicated into the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> SpawnPoolProcessRunner:
        _spawn_start_method_check()
        return SpawnPoolProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )


@dataclass
class PoolRunnerMemory:
    context: LabContext
    storage: Storage


_RUNNER_FORK_POOL_MEMORY: dict[UUID, PoolRunnerMemory] = {}


class ForkPoolProcessRunner(PoolProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> None:
        super().__init__(
            mp_context=multiprocessing.get_context('fork'),
            max_workers=max_workers,
        )
        self.uuid = uuid4()
        _RUNNER_FORK_POOL_MEMORY[self.uuid] = PoolRunnerMemory(
            context=context,
            storage=storage,
        )

    @staticmethod
    def _fork_task_subprocess_func(*, task: Task, task_name: str, use_cache: bool,
                                   results_map: ResultsMap, task_event_queue: Queue,
                                   log_queue: Queue, uuid: UUID) -> TaskResult:
        runner_memory = _RUNNER_FORK_POOL_MEMORY[uuid]
        return _task_subprocess_func(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            filtered_context=task.filter_context(runner_memory.context),
            storage=runner_memory.storage,
            results_map=results_map,
            task_event_queue=task_event_queue,
            log_queue=log_queue,
        )

    def _schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool) -> ConcurrentFuture:
        results_map: dict[Task, TaskResult] = {}
        if not use_cache:
            # In order to minimise memory use, only transfer results
            # to the subprocess if we are going to run the task (and
            # not just load its result from cache).
            results_map = {
                dependency_task: self.results_map[dependency_task]
                for dependency_task in get_direct_dependencies(task, all_identities=False)
            }
        return self.concurrent_executor.submit(
            self._fork_task_subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map=results_map,
            task_event_queue=self.task_event_queue,
            log_queue=self.log_queue,
            uuid=self.uuid,
        )

    def _close_executor(self) -> None:
        super()._close_executor()
        try:
            del _RUNNER_FORK_POOL_MEMORY[self.uuid]
        except KeyError:
            # uuid not may be found if _close_executor() is called twice.
            pass


class ForkPoolRunnerBackend(RunnerBackend):
    """Runner Backend that runs tasks on a pool of forked subprocesses.

    The context is shared in-memory between each subprocess. Dependency task
    results are copied/duplicated into the memory of each subprocess.

    Because process forking is more efficient than spawning, and
    because of the added benefit of not duplicating the context for
    each task, this runner backend is recommended for any system that
    supports process forking.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ForkPoolProcessRunner:
        _fork_start_method_check()
        return ForkPoolProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )


# === Subprocesses Per-Task  ===

class PerTaskProcessRunner(ProcessRunner[ExecutorFuture], ABC):

    def __init__(self, *, mp_context: BaseContext, max_workers: int | None) -> None:
        super().__init__()
        self.executor = ProcessExecutor(
            mp_context=mp_context,
            max_workers=max_workers,
        )

    def _get_completed_futures(self, futures: list[ExecutorFuture], timeout_seconds: float | None) -> list[ExecutorFuture]:
        done, _ = self.executor.wait(futures, timeout_seconds=timeout_seconds)
        return done

    def cancel(self) -> None:
        self.executor.cancel()

    def stop(self) -> None:
        self.executor.stop()

    def _close_executor(self) -> None:
        pass


@dataclass
class PerTaskRunnerMemory:
    context: LabContext
    storage: Storage
    results_map: ResultsMap


_RUNNER_FORK_PER_TASK_MEMORY: dict[UUID, PerTaskRunnerMemory] = {}


class ForkPerTaskProcessRunner(PerTaskProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> None:
        super().__init__(
            mp_context=multiprocessing.get_context('fork'),
            max_workers=max_workers,
        )
        self.uuid = uuid4()
        _RUNNER_FORK_PER_TASK_MEMORY[self.uuid] = PerTaskRunnerMemory(
            context=context,
            storage=storage,
            results_map=self.results_map,
        )

    @staticmethod
    def _fork_task_subprocess_func(*, task: Task, task_name: str, use_cache: bool,
                                   task_event_queue: Queue, log_queue: Queue,
                                   uuid: UUID) -> TaskResult:
        runner_memory = _RUNNER_FORK_PER_TASK_MEMORY[uuid]
        return _task_subprocess_func(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            filtered_context=task.filter_context(runner_memory.context),
            storage=runner_memory.storage,
            results_map=runner_memory.results_map,
            task_event_queue=task_event_queue,
            log_queue=log_queue,
        )

    def _schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool) -> ExecutorFuture:
        return self.executor.submit(
            self._fork_task_subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            task_event_queue=self.task_event_queue,
            log_queue=self.log_queue,
            uuid=self.uuid,
        )

    def _close_executor(self) -> None:
        super()._close_executor()
        try:
            del _RUNNER_FORK_PER_TASK_MEMORY[self.uuid]
        except KeyError:
            # uuid not may be found if _close_executor() is called twice.
            pass


class ForkPerTaskRunnerBackend(RunnerBackend):
    """Runner Backend that runs each task in a separate forked
    subprocess.

    The context and dependency task results are shared in-memory
    between each subprocess but at the cost of forking a new
    subprocess for each task.

    This runner backend is best used when dependency task results are
    large (so time will be saved through memory sharing) compared to
    the overall number of tasks (for large numbers of tasks, forking a
    separate process for each may be a substantial overhead).

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ForkPerTaskProcessRunner:
        _fork_start_method_check()
        return ForkPerTaskProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )
