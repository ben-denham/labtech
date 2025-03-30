"""Core types of labtech."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from inspect import isclass
from typing import (
    IO,
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
)


@dataclass(frozen=True)
class TaskInfo:
    orig_post_init: Optional[Callable]
    cache: 'Cache'
    max_parallel: Optional[int]
    mlflow_run: bool


CovariantResultT = TypeVar('CovariantResultT', covariant=True)
ResultT = TypeVar('ResultT')
"""Type variable for result returned by the `run` method of a
[`Task`][labtech.types.Task]."""
LabContext = dict[str, Any]


@dataclass(frozen=True)
class ResultMeta:
    """Metadata about the execution of a task. If the task is loaded from
    cache, the metadata is also loaded from the cache."""
    start: Optional[datetime]
    """The timestamp when the task's execution began."""
    duration: Optional[timedelta]
    """The time that the task took to execute."""


@dataclass(frozen=True)
class TaskResult(Generic[ResultT]):
    value: ResultT
    meta: ResultMeta


class ResultsMap(Protocol, Generic[ResultT]):

    def __getitem__(self, task: 'Task[ResultT]') -> TaskResult[ResultT]:
        pass

    def get(self, task: 'Task[ResultT]') -> Optional[TaskResult[ResultT]]:
        pass

    def __contains__(self, task: 'Task[ResultT]') -> bool:
        pass


@dataclass
class Task(Protocol, Generic[CovariantResultT]):
    """Interface provided by any class that is decorated by
    [`labtech.task`][labtech.task]."""
    _lt: TaskInfo
    _is_task: Literal[True]
    _results_map: Optional[ResultsMap]
    cache_key: str
    """The key that uniquely identifies the location for this task within cache storage."""
    context: Optional[LabContext]
    """Context variables from the Lab that can be accessed when the task is running."""
    result_meta: Optional[ResultMeta]
    """Metadata about the execution of the task."""

    def _set_results_map(self, results_map: ResultsMap):
        pass

    def _set_result_meta(self, result_meta: ResultMeta):
        pass

    @property
    def result(self) -> CovariantResultT:
        """Returns the result executed/loaded for this task. If no result is
        available in memory, accessing this property raises a `TaskError`."""

    def set_context(self, context: LabContext):
        """Set the context that is made available to the task while it is
        running."""

    def filter_context(self, context: LabContext) -> LabContext:
        """User-overridable method to filter/transform the context to
        be provided to the task. The default implementation provides
        the full context to the task. The filtering may take into
        account the values of the task's attributes.

        This can be useful for selecting subsets of large contexts in
        order to reduce data transferred to non-forked subprocesses or
        other kinds of processes in parallel processing frameworks.

        """

    def run(self):
        """User-provided method that executes the task parameterised by the
        attributes of the task.

        Usually executed by [`Lab.run_tasks()`][labtech.Lab.run_tasks]
        instead of being called directly.

        """


TaskT = TypeVar("TaskT", bound=Task)


def is_task_type(cls):
    """Returns `True` if the given `cls` is a class decorated with
    [`labtech.task`][labtech.task]."""
    return isclass(cls) and isinstance(getattr(cls, '_lt', None), TaskInfo)


def is_task(obj):
    """Returns `True` if the given `obj` is an instance of a task class."""
    return is_task_type(type(obj)) and hasattr(obj, '_is_task')


class Storage(ABC):
    """Storage provider for persisting cached task results."""

    @abstractmethod
    def find_keys(self) -> Sequence[str]:
        """Returns the keys of all currently cached task results."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Returns `True` if the given task `key` is present in the storage
        cache."""

    @abstractmethod
    def file_handle(self, key: str, filename: str, *, mode: str = 'r') -> IO:
        """Opens and returns a File-like object for a single file within the
        storage cache.

        Args:
            key: The task key of the cached result containing the file.
            filename: The name of the file to open.
            mode: The file mode to open the file with.

        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Deletes the cached result for the task with the given `key`."""


class Cache(ABC):
    """Cache that controls saving task results into
    [Storage][labtech.storage.Storage] providers."""

    @abstractmethod
    def cache_key(self, task: Task) -> str:
        """Returns the key to identify the given `task` within this cache."""

    @abstractmethod
    def is_cached(self, storage: Storage, task: Task) -> bool:
        """Returns `True` if a result is cached for the given `task` within
        the given `storage`."""

    @abstractmethod
    def save(self, storage: Storage, task: Task[ResultT], result: TaskResult[ResultT]) -> None:
        """Save the given `result` for the given `task` into the given
        `storage`."""

    @abstractmethod
    def load_task(self, storage: Storage, task_type: Type[TaskT], key: str) -> TaskT:
        """Loads the task instance of the given `task_type` for the given
        `key` from the given `storage`.

        The task's result is not loaded by this method.
        """

    @abstractmethod
    def load_result_with_meta(self, storage: Storage, task: Task[ResultT]) -> TaskResult[ResultT]:
        """Loads the result and metadata for the given task from the storage
        provider.

        Args:
            storage: Storage provider to load the result from
            task: task instance to load the result for

        """

    @abstractmethod
    def delete(self, storage: Storage, task: Task) -> None:
        """Deletes the cached result for the given `task` from the given
        `storage`."""


TaskMonitorInfoValue = datetime | str | int | float
TaskMonitorInfoItem = TaskMonitorInfoValue | tuple[TaskMonitorInfoValue, str]
TaskMonitorInfo = dict[str, TaskMonitorInfoItem]


class Runner(ABC):
    """Manages the execution of [Tasks][labtech.types.Task], typically
    by delegating to a parallel processing framework."""

    @abstractmethod
    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        """
        Args:
            context: A dictionary of additional variables made available to
                tasks. It is the responsibility of the Runner to ensure a ta.
            storage: Where task results should be cached to.
            max_workers: The maximum number of parallel worker processes for
                running tasks.
        """

    @abstractmethod
    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        """Submit the given task object for execution.

        The implementation of this method must transform the cotnext
        passed to the task with `task.filter_context()`.

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
        submitted tasks is done, then return a list of tasks in a done
        state and a list of tasks in all other states.

        Each task is returned as a pair where the first value is the
        task itself, and the second value is either:

        * For a successfully completed task: Metadata of the result.
        * For a failed or cancelled task: The exception that was raised.

        """

    @abstractmethod
    def close(self, *, wait: bool) -> None:
        """Halt execution of all tasks and clean up resources. If
        wait=True, wait for currently running tasks to finish."""

    @abstractmethod
    def submitted_task_count(self) -> int:
        """Returns the number of tasks that have been submitted but
        not yet returned from a call to wait()."""

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
        """Returns monitoring information about all tasks that are
        currently being executed."""


class RunnerBackend(ABC):

    @abstractmethod
    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> Runner:
        """Return a Runner prepared with the given configuration."""
