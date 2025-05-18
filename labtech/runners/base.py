from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime
from enum import Enum
from typing import Any, Iterator, Optional, Sequence

from frozendict import frozendict

from labtech.exceptions import LabError
from labtech.params import ParamHandlerManager
from labtech.tasks import is_task
from labtech.types import LabContext, ResultMeta, Storage, Task, TaskMonitorInfo, TaskResult
from labtech.utils import logger


class Runner(ABC):
    """Manages the execution of [Tasks][labtech.types.Task], typically
    by delegating to a parallel processing framework."""

    @abstractmethod
    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        """Submit the given task object to be run and have its result cached.

        It is up to the Runner to decide when to start running the
        task (i.e. when resources become available).

        The implementation of this method should run the task by
        effectively calling:

        ```
        # param_handler_manager needs to be instantiated in remote processes
        # that don't inherit from the main process:
        param_handler_manager.instantiate()

        for dependency_task in get_direct_dependencies(task, all_identities=True):
            # Where results_map is expected to contain the TaskResult for
            # each dependency_task.
            dependency_task._set_results_map(results_map)

        current_process = multiprocessing.current_process()
        orig_process_name = current_process.name
        try:
            # If the thread name or similar is set instead of the process
            # name, then the Runner should update the handler of the global
            # labtech.utils.logger to include that instead of the process name.
            current_process.name = task_name
            return labtech.runners.base.run_or_load_task(
                task=task,
                use_cache=use_cache,
                filtered_context=task.filter_context(context),
                storage=storage,
            )
        finally:
            current_process.name = orig_process_name
        ```

        Args:
            task: The task to execute.
            task_name: Name to use when referring to the task in logs.
            use_cache: If True, the task's result should be fetched from the
                cache if it is available (fetching should still be done in a
                delegated process).

        """

    @abstractmethod
    def wait(self, *, timeout_seconds: Optional[float]) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        """Wait up to timeout_seconds or until at least one of the
        submitted tasks is done, then return an iterator of tasks in a
        done state and a list of tasks in all other states.

        Each task is returned as a pair where the first value is the
        task itself, and the second value is either:

        * For a successfully completed task: Metadata of the result.
        * For a task that fails with any BaseException descendant: The exception
          that was raised.

        Cancelled tasks are never returned.

        """

    @abstractmethod
    def cancel(self) -> None:
        """Cancel all submitted tasks that have not yet been started."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all currently running tasks."""

    @abstractmethod
    def close(self) -> None:
        """Clean up any resources used by the Runner after all tasks
        are finished, cancelled, or stopped."""

    @abstractmethod
    def pending_task_count(self) -> int:
        """Returns the number of tasks that have been submitted but
        not yet cancelled or returned from a call to wait()."""

    @abstractmethod
    def get_result(self, task: Task) -> TaskResult:
        """Returns the in-memory result for a task that was
        successfully run by this Runner. Raises a KeyError for a
        result with no in-memory result."""

    @abstractmethod
    def remove_results(self, tasks: Sequence[Task]) -> None:
        """Removes the in-memory results for tasks that were
        sucessfully run by this Runner. Ignores tasks that have no
        in-memory result."""

    @abstractmethod
    def get_task_infos(self) -> list[TaskMonitorInfo]:
        """Returns a snapshot of monitoring information about each
        task that is currently running."""


class RunnerBackend(ABC):
    """Factory class to construct [Runner][labtech.runners.Runner] objects."""

    @abstractmethod
    def build_runner(self, *, context: LabContext, storage: Storage,
                     param_handler_manager: ParamHandlerManager,
                     max_workers: Optional[int]) -> Runner:
        """Return a Runner prepared with the given configuration.

        Args:
            context: Additional variables made available to tasks that aren't
                considered when saving to/loading from the cache.
            storage: Where task results should be cached to.
            param_handler_manager: Custom parameter handling configuration
                to be instantiated on remote processes.
            max_workers: The maximum number of parallel worker processes for
                running tasks.
        """


@contextmanager
def optional_mlflow(task: Task):
    """Context manager to set mlflow "run" configuration for a task if
    mlflow_run=True is set for the task type."""

    def log_params(value: Any, *, path: str = ''):
        prefix = path if path == '' else f'{path}.'
        if is_task(value):
            for field in fields(value):
                log_params(getattr(value, field.name), path=f'{prefix}{field.name}')
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                log_params(item, path=f'{prefix}{i}')
        elif isinstance(value, frozendict):
            for key, item in value.items():
                log_params(item, path=f'{prefix}{key}')
        elif isinstance(value, Enum):
            mlflow.log_param(path, f'{type(value).__qualname__}.{value.name}')
        elif ((value is None)
              or isinstance(value, str)
              or isinstance(value, bool)
              or isinstance(value, float)
              or isinstance(value, int)):
            mlflow.log_param(path, value)
        else:
            raise LabError(
                (f"Unable to mlflow log parameter '{path}' of type '{type(value).__qualname__}' "
                 f"in task of type '{type(task).__qualname__}'.")
            )

    if task._lt.mlflow_run:
        try:
            import mlflow
        except ImportError:
            raise LabError(
                (f"Task type '{type(task).__qualname__}' is configured with mlflow_run=True, but "
                 "mlflow cannot be imported. You can install mlflow with `pip install mlflow`.")
            )
        with mlflow.start_run():
            mlflow.set_tag('labtech_task_type', type(task).__qualname__)
            log_params(task)
            yield
    else:
        yield


def run_or_load_task(task: Task, use_cache: bool, filtered_context: LabContext, storage: Storage) -> TaskResult:
    """Called by a Runner to either:

    1. Run the task with the given filtered_context and cache its result in the given
       storage and return the result OR
    2. Load and return the task's result from the cache if it is present
       and use_cache=True

    """
    if use_cache:
        logger.debug(f"Loading from cache: '{task}'")
        task_result = task._lt.cache.load_result_with_meta(storage, task)
        return task_result
    else:
        logger.debug(f"Running: '{task}'")
        task.set_context(filtered_context)
        with optional_mlflow(task):
            start = datetime.now()
            result = task.run()
            end = datetime.now()
        task_result = TaskResult(
            value=result,
            meta=ResultMeta(
                start=start,
                duration=(end - start),
            ),
        )
        task._lt.cache.save(storage, task, task_result)
        logger.debug(f"Completed: '{task}'")
        return task_result
