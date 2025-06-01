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


FutureT = TypeVar('FutureT', bound=ExecutorFuture | ConcurrentFuture)


class ProcessManager(ABC, Generic[FutureT]):

    def __init__(self) -> None:
        self.results_map: dict[Task, TaskResult] = {}

    def set_result(self, task: Task, task_result: TaskResult) -> None:
        self.results_map[task] = task_result

    def get_result(self, task: Task) -> TaskResult:
        return self.results_map[task]

    def remove_result(self, task: Task) -> None:
        if task not in self.results_map:
            return
        logger.debug(f"Removing result from in-memory cache for task: '{task}'")
        del self.results_map[task]

    @abstractmethod
    def schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool,
                            task_event_queue: Queue, log_queue: Queue) -> FutureT:
        """Should submit the execution of _task_subprocess_func() on the given
        in a subprocess and return the resulting future.

        The implementation of this method to load or otherwise prepare
        context or dependency results for the task.

        """

    @abstractmethod
    def get_completed_futures(self, futures: list[FutureT], timeout_seconds: float | None) -> list[FutureT]:
        """Return a sub-sequence of the given futures that have been completed."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel all scheduled subprocesses that have not yet been started."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all currently running subprocesses."""

    @abstractmethod
    def close(self) -> None:
        """Stop all currently running subprocesses."""


class ProcessRunner(Runner, Generic[FutureT]):
    """Runner based on Python multiprocessing."""

    def __init__(self, *, process_manager: ProcessManager[FutureT]) -> None:
        self.process_manager = process_manager

        self.log_queue = multiprocessing.Manager().Queue(-1)
        self.task_event_queue = multiprocessing.Manager().Queue(-1)
        self.process_monitor = ProcessMonitor(task_event_queue = self.task_event_queue)

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
        future = self.process_manager.schedule_subprocess(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            task_event_queue=self.task_event_queue,
            log_queue=self.log_queue,
        )
        self.future_to_task[future] = task

    def wait(self, *, timeout_seconds: float | None) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        self._consume_log_queue()
        done = self.process_manager.get_completed_futures(
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
                self.process_manager.set_result(task, task_result)
                yield (task, task_result.meta)
        self.future_to_task = {
            future: self.future_to_task[future]
            for future in self.future_to_task
            if future not in done
        }

    def cancel(self) -> None:
        self.process_manager.cancel()

    def stop(self) -> None:
        self.process_manager.stop()

    def close(self) -> None:
        self._consume_log_queue()
        self.process_manager.close()

    def pending_task_count(self) -> int:
        return len(self.future_to_task)

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        return self.process_monitor.get_process_infos()

    def get_result(self, task: Task) -> TaskResult:
        return self.process_manager.get_result(task)

    def remove_results(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            self.process_manager.remove_result(task)


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

class PoolProcessManager(ProcessManager[ConcurrentFuture], ABC):

    def __init__(self, *, mp_context: BaseContext, max_workers: int | None) -> None:
        super().__init__()
        self.concurrent_executor = ProcessPoolExecutor(
            mp_context=mp_context,
            max_workers=max_workers,
        )

    def get_completed_futures(self, futures: list[ExecutorFuture], timeout_seconds: float | None) -> list[ExecutorFuture]:
        done, _ = wait_futures(futures, timeout=timeout_seconds, return_when=FIRST_COMPLETED)
        return done

    def cancel(self) -> None:
        self.concurrent_executor.shutdown(wait=True, cancel_futures=True)

    def stop(self) -> None:
        self.concurrent_executor.shutdown(wait=True, cancel_futures=True)
        for process in self.concurrent_executor._processes.values():
            process.terminate()

    def close(self) -> None:
        self.concurrent_executor.shutdown(wait=True)


class SpawnPoolProcessManager(PoolProcessManager):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> None:
        super().__init__(
            mp_context=multiprocessing.get_context('spawn'),
            max_workers=max_workers,
        )
        self.context = context
        self.storage = storage

    def schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool,
                            task_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
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
            task_event_queue=task_event_queue,
            log_queue=log_queue,
        )


class SpawnPoolRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs tasks on a pool of spawned subprocesses.

    The required context and dependency task results are
    copied/duplicated into the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ProcessRunner:
        _spawn_start_method_check()
        return ProcessRunner(
            process_manager=SpawnPoolProcessManager(
                context=context,
                storage=storage,
                max_workers=max_workers,
            )
        )


@dataclass
class PoolRunnerMemory:
    context: LabContext
    storage: Storage


_RUNNER_FORK_POOL_MEMORY: dict[UUID, PoolRunnerMemory] = {}


class ForkPoolProcessManager(PoolProcessManager):

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
                                   log_queue: Queue, uuid: UUID):
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

    def schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool,
                            task_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
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
            task_event_queue=task_event_queue,
            log_queue=log_queue,
            uuid=self.uuid,
        )

    def close(self) -> None:
        super().close()
        try:
            del _RUNNER_FORK_POOL_MEMORY[self.uuid]
        except KeyError:
            # uuid not may be found if close() is called twice.
            pass


class ForkPoolRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs tasks on a pool of forked subprocesses.

    The context is shared in-memory between each subprocess. Dependency task
    results are copied/duplicated into the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ProcessRunner:
        _fork_start_method_check()
        return ProcessRunner(
            process_manager=ForkPoolProcessManager(
                context=context,
                storage=storage,
                max_workers=max_workers,
            )
        )


# === Subprocesses Per-Task  ===

class PerTaskProcessManager(ProcessManager[ExecutorFuture]):

    def __init__(self, *, mp_context: BaseContext, max_workers: int | None) -> None:
        super().__init__()
        self.executor = ProcessExecutor(
            mp_context=mp_context,
            max_workers=max_workers,
        )

    def get_completed_futures(self, futures: list[ExecutorFuture], timeout_seconds: float | None) -> list[ExecutorFuture]:
        done, _ = self.executor.wait(futures, timeout_seconds=timeout_seconds)
        return done

    def cancel(self) -> None:
        self.executor.cancel()

    def stop(self) -> None:
        self.executor.stop()

    def close(self) -> None:
        pass


class SpawnPerTaskProcessManager(PerTaskProcessManager):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> None:
        super().__init__(
            mp_context=multiprocessing.get_context('spawn'),
            max_workers=max_workers,
        )
        self.context = context
        self.storage = storage

    def schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool,
                            task_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
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

        return self.executor.submit(
            _task_subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map=results_map,
            filtered_context=filtered_context,
            storage=self.storage,
            task_event_queue=task_event_queue,
            log_queue=log_queue,
        )


class SpawnPerTaskRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs each task in a spawned subprocess.

    The required context and dependency task results are
    copied/duplicated into the memory of each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ProcessRunner:
        _spawn_start_method_check()
        return ProcessRunner(
            process_manager=SpawnPerTaskProcessManager(
                context=context,
                storage=storage,
                max_workers=max_workers,
            )
        )


@dataclass
class PerTaskRunnerMemory:
    context: LabContext
    storage: Storage
    results_map: ResultsMap


_RUNNER_FORK_PER_TASK_MEMORY: dict[UUID, PerTaskRunnerMemory] = {}


class ForkPerTaskProcessManager(PerTaskProcessManager):

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
                                   uuid: UUID):
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

    def schedule_subprocess(self, *, task: Task, task_name: str, use_cache: bool,
                            task_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
        return self.executor.submit(
            self._fork_task_subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            task_event_queue=task_event_queue,
            log_queue=log_queue,
            uuid=self.uuid,
        )

    def close(self) -> None:
        super().close()
        try:
            del _RUNNER_FORK_PER_TASK_MEMORY[self.uuid]
        except KeyError:
            # uuid not may be found if close() is called twice.
            pass


class ForkPerTaskRunnerBackend(RunnerBackend):
    """
    Runner Backend that runs each task in a forked subprocess.

    The context and dependency task results are shared in-memory
    between each subprocess.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> ProcessRunner:
        _fork_start_method_check()
        return ProcessRunner(
            process_manager=ForkPerTaskProcessManager(
                context=context,
                storage=storage,
                max_workers=max_workers,
            )
        )
