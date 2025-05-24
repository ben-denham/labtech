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
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

# Type to represent any value that can be handled by Python's default
# json encoder and decoder.
jsonable = Union[None, str, bool, float, int,
                 dict[str, 'jsonable'], list['jsonable']]


@dataclass(frozen=True)
class TaskInfo:
    orig_post_init: Optional[Callable]
    cache: 'Cache'
    current_code_version: Optional[str]
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
    _cache_key: Optional[str]
    context: Optional[LabContext]
    """Context variables from the Lab that can be accessed when the task is running."""
    result_meta: Optional[ResultMeta]
    """Metadata about the execution of the task."""
    code_version: Optional[str]
    """Identifier for the version of task's implementation. If this task was
    loaded from cache, it may have a different value to that currently specified
    in the decorator."""

    def _set_results_map(self, results_map: ResultsMap):
        pass

    def _set_result_meta(self, result_meta: ResultMeta):
        pass

    @property
    def current_code_version(self) -> Optional[str]:
        """Identifier for the current version of task's implementation
        as specified in the [`labtech.task`][labtech.task] decorator."""

    @property
    def cache_key(self) -> str:
        """The key that uniquely identifies the location for this task
        within cache storage."""

    @property
    def result(self) -> CovariantResultT:
        """The result executed/loaded for this task. If no result is
        available in memory, accessing this property raises a `TaskError`."""

    def set_context(self, context: LabContext):
        """Set the context that is made available to the task while it is
        running."""

    def runner_options(self) -> dict[str, Any]:
        """User-overridable method to a dictionary of options to
        further control the behaviour of specific types of runner
        backend - refer to the documentation of each runner backend
        for supported options. The implementation may make use of the
        task's parameter values.

        """

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

    def __hash__(self) -> int: ...

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


class Serializer(ABC):
    """Serializer for producing serialized JSON representations of
    Task objects, and deserializing JSON back into Task objects."""

    @abstractmethod
    def serialize_task(self, task: Task) -> dict[str, jsonable]:
        """Convert the given task into a JSON-compatible
        representation composed only of dictionaries, lists, strings,
        numbers and `None`."""

    @abstractmethod
    def deserialize_task(self, serialized: dict[str, jsonable], *, result_meta: Optional[ResultMeta]) -> Task:
        """Convert the given serialized representation returned by
        serialize_task() back into the original task."""

    @abstractmethod
    def serialize_value(self, value: Any) -> jsonable:
        """Convert the given value into a JSON-compatible
        representation composed only of dictionaries, lists, strings,
        numbers and `None`."""

    @abstractmethod
    def deserialize_value(self, value: jsonable):
        """Convert the given serialized representation returned by
        serialize_value() back into the original value."""

    @abstractmethod
    def serialize_class(self, cls: Type) -> jsonable:
        """Convert the given class into a string representation."""

    @abstractmethod
    def deserialize_class(self, serialized_class: jsonable) -> Type:
        """Load the class named in the given serialized representation
        returned by serialize_class()."""


@runtime_checkable
class ParamHandler(Protocol):
    """Protocol for custom parameter handlers that can define how
    Labtech should handle the processing, serialization, and
    deserialization of additional parameter types."""

    def handles(self, value: Any) -> bool:
        """Returns True if the given parameter value should be handled
        by this class."""

    def find_tasks(self, value: Any, *, find_tasks_in_param: Callable[[Any], Sequence[Task]]) -> list[Task]:
        """Given a parameter value, return all tasks within it (not
        including tasks within those tasks).

        The provided `find_tasks_in_param` should be called to find
        tasks in anynested elements within the value."""

    def serialize(self, value: Any, *, serializer: Serializer) -> jsonable:
        """Convert the given parameter value into a JSON-compatible
        representation composed only of dictionaries, lists, strings,
        numbers and `None`.

        Also receives the full Serializer, which can be used to call
        `serializer.serialize_value()` to serialize nested elements
        within the value."""

    def deserialize(self, serialized: jsonable, *, serializer: Serializer) -> Any:
        """Convert the given serialized representation returned by
        serialize() back into the original parameter value.

        Also receives the full Serializer, which can be used to call
        `serializer.deserialize_value()` to deserialize nested elements
        within the serialized representation."""


TaskMonitorInfoValue = datetime | str | int | float
TaskMonitorInfoItem = TaskMonitorInfoValue | tuple[TaskMonitorInfoValue, str]
TaskMonitorInfo = dict[str, TaskMonitorInfoItem]
