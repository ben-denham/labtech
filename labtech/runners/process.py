from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging.handlers import QueueHandler
from queue import Empty
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import psutil

from labtech.exceptions import RunnerError
from labtech.monitor import get_process_info
from labtech.tasks import get_direct_dependencies
from labtech.types import Runner, RunnerBackend
from labtech.utils import LoggerFileProxy, get_supported_start_methods, is_interactive, logger

from ._process_executor import ProcessExecutor
from .base import run_or_load_task

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from multiprocessing.context import BaseContext, SpawnContext
    from queue import Queue
    from uuid import UUID

    from labtech.types import LabContext, ResultMeta, ResultsMap, Storage, Task, TaskMonitorInfo, TaskResult

    from ._process_executor import ExecutorFuture

    if sys.platform != 'win32':
        from multiprocessing.context import ForkContext
    else:
        ForkContext = BaseContext


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
        mp_context = self._get_mp_context()
        self.process_event_queue = mp_context.Manager().Queue(-1)
        self.process_monitor = ProcessMonitor(process_event_queue = self.process_event_queue)
        self.log_queue = multiprocessing.Manager().Queue(-1)
        self.executor = ProcessExecutor(
            mp_context=mp_context,
            max_workers=max_workers,
        )

        self.results_map: dict[Task, TaskResult] = {}
        self.future_to_task: dict[ExecutorFuture, Task] = {}

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
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
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
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

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
        self._consume_log_queue()

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
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
        """Should submit the execution of self._subprocess_func() on the given
        task to the given executor and return the resulting ExecutorFuture.

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
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
        if is_interactive() and task.__class__.__module__ == '__main__':
            raise RunnerError(
                (f'Unable to submit {task.__class__.__qualname__} tasks to '
                 'SpawnProcessRunner because the task type is defined in the '
                 '__main__ module from an interactive Python session. '
                 'Please define your task types in a separate `.py` Python '
                 'module file. For details, see: '
                 'https://ben-denham.github.io/labtech/cookbook/#spawn-interactive-main')
            )

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
                     use_cache: bool, process_event_queue: Queue, log_queue: Queue) -> ExecutorFuture:
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
        super().close()
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
