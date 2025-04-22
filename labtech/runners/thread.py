import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from importlib import import_module
from typing import Iterator, Optional, Sequence

import psutil

from labtech.exceptions import RunnerError
from labtech.monitor import get_process_info
from labtech.runners.base import run_or_load_task
from labtech.tasks import get_direct_dependencies
from labtech.types import LabContext, ResultMeta, Runner, RunnerBackend, Storage, Task, TaskMonitorInfo, TaskResult
from labtech.utils import OrderedSet, logger


class KillThread(Exception):
    pass


@dataclass(frozen=True)
class TaskInfo:
    task: Task
    task_name: str
    use_cache: bool


class ThreadRunner(Runner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> None:
        self.context = context
        self.storage = storage
        self.max_workers = (os.cpu_count() or 1) + 4 if max_workers is None else max_workers

        self.results_map: dict[Task, TaskResult] = {}
        self.future_to_task: dict[Future, Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Used for task monitoring
        self.active_task_infos: OrderedSet[TaskInfo] = OrderedSet()
        self.current_process = psutil.Process()
        self.child_processes: dict[int, psutil.Process] = {}

    def _thread_func(self, task: Task, task_name: str, use_cache: bool) -> TaskResult:
        task_info = TaskInfo(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
        )
        self.active_task_infos.add(task_info)
        try:
            for dependency_task in get_direct_dependencies(task):
                dependency_task._set_results_map(self.results_map)
            return run_or_load_task(
                task=task,
                task_name=task_name,
                use_cache=use_cache,
                filtered_context=task.filter_context(self.context),
                storage=self.storage,
            )
        finally:
            self.active_task_infos.remove(task_info)

    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        future = self.executor.submit(
            self._thread_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
        )
        self.future_to_task[future] = task

    def wait(self, *, timeout_seconds: Optional[float]) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        done, _ = wait(
            self.future_to_task.keys(),
            timeout=timeout_seconds,
            return_when=FIRST_COMPLETED,
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

    def cancel(self) -> None:
        for future in self.future_to_task.keys():
            future.cancel()

    def stop(self) -> None:
        try:
            ctypes = import_module('ctypes')
        except ImportError:
            raise RunnerError((
                'Failed to kill running threads because ctypes could not be imported. '
                "Terminating the 'thread' runner is not currently supported on Python runtimes other than CPython."
            ))
        for thread in self.executor._threads:
            # Raise an exception in each thread:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident),
                ctypes.py_object(KillThread),
            )

    def close(self) -> None:
        self.executor.shutdown(wait=True)

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
        process_info, self.child_processes = get_process_info(
            self.current_process,
            previous_child_processes=self.child_processes,
            name='N/A',
            status='N/A',
        )
        if process_info is None:
            return []
        return [
            {
                **process_info,
                'name': task_info.task_name,
                'status': ('loading' if task_info.use_cache else 'running'),
            }
            for task_info in self.active_task_infos
        ]


class ThreadRunnerBackend(RunnerBackend):
    """Runner Backend that runs tasks asynchronously in separate
    threads.

    Memory use is reduced by sharing the same in-memory context and
    dependency task results across threads.

    """

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> ThreadRunner:
        return ThreadRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )
