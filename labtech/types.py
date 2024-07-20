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


class Runner(ABC):
    """TODO"""

    @abstractmethod
    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        pass

    @abstractmethod
    def start_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        """TODO"""

    @abstractmethod
    def pending_task_count(self) -> int:
        """TODO"""

    @abstractmethod
    def wait(self) -> Iterator[tuple[Task, ResultMeta | Exception]]:
        """TODO"""

    @abstractmethod
    def cancel(self) -> None:
        """TODO"""

    @abstractmethod
    def terminate(self) -> None:
        """TODO"""

    @abstractmethod
    def close(self, *, wait: bool) -> None:
        """TODO"""

    @abstractmethod
    def get_result(self, task: Task) -> TaskResult:
        """TODO"""

    @abstractmethod
    def remove_result(self, task: Task) -> None:
        """TODO"""


class RunnerBackend(ABC):

    @abstractmethod
    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> Runner:
        """TODO"""
