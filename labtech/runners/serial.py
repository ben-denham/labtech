from __future__ import annotations

import multiprocessing
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil

from labtech.monitor import get_process_info
from labtech.tasks import get_direct_dependencies
from labtech.types import Runner, RunnerBackend
from labtech.utils import logger

from .base import run_or_load_task

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from labtech.types import LabContext, ResultMeta, Storage, Task, TaskMonitorInfo, TaskResult


@dataclass(frozen=True)
class TaskSubmission:
    task: Task
    task_name: str
    use_cache: bool


class SerialRunner(Runner):

    def __init__(self, *, context: LabContext, storage: Storage):
        self.context = context
        self.storage = storage
        self.task_submissions: deque[TaskSubmission] = deque()
        self.results_map: dict[Task, TaskResult] = {}

        # Used for task monitoring
        self.current_process = psutil.Process()
        self.child_processes: dict[int, psutil.Process] = {}

    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        self.task_submissions.append(TaskSubmission(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
        ))

    def wait(self, *, timeout_seconds: float | None) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        try:
            task_submission = self.task_submissions.popleft()
        except IndexError:
            return

        task = task_submission.task
        try:
            for dependency_task in get_direct_dependencies(task, all_identities=True):
                dependency_task._set_results_map(self.results_map)

            current_process = multiprocessing.current_process()
            orig_process_name = current_process.name
            try:
                current_process.name = task_submission.task_name
                task_result = run_or_load_task(
                    task=task,
                    use_cache=task_submission.use_cache,
                    filtered_context=task.filter_context(self.context),
                    storage=self.storage,
                )
            finally:
                current_process.name = orig_process_name
        except KeyboardInterrupt:
            raise
        except BaseException as ex:
            yield (task, ex)
        else:
            self.results_map[task] = task_result
            yield (task, task_result.meta)

    def cancel(self) -> None:
        self.task_submissions.clear()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def pending_task_count(self) -> int:
        return len(self.task_submissions)

    def get_result(self, task: Task) -> TaskResult:
        return self.results_map[task]

    def remove_results(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            if task not in self.results_map:
                return
            logger.debug(f"Removing result from in-memory cache for task: '{task}'")
            del self.results_map[task]

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        try:
            next_task_submission = self.task_submissions[0]
        except IndexError:
            return []

        # Get info about current process (which is running the tasks serially).
        process_info, self.child_processes = get_process_info(
            self.current_process,
            previous_child_processes=self.child_processes,
            name=next_task_submission.task_name,
            status=('loading' if next_task_submission.use_cache else 'running'),
        )
        if process_info is None:
            return []
        return [process_info]


class SerialRunnerBackend(RunnerBackend):
    """Runner Backend that runs each task serially in the main process
    and thread."""

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: int | None) -> SerialRunner:
        return SerialRunner(
            context=context,
            storage=storage,
        )
