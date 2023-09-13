"""Core types of labtech."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import isclass
from typing import Any, Callable, Dict, IO, Literal, Optional, Protocol, Sequence, Type


@dataclass(frozen=True)
class TaskInfo:
    orig_post_init: Optional[Callable]
    cache: 'Cache'
    max_parallel: Optional[int]


ResultsMap = Dict['Task', Any]


class Task(Protocol):
    """Interface provided by any class that is decorated by
    [`labtech.task`][labtech.task]."""
    _lt: TaskInfo
    _is_task: Literal[True]
    _results_map: Optional[ResultsMap]
    cache_key: str
    """The key that uniquely identifies the location for this task within cache storage."""

    def _set_results_map(self, results_map: ResultsMap):
        pass

    @property
    def result(self) -> Any:
        """Returns the result executed/loaded for this task. If no result is
        available in memory, accessing this property raises a `TaskError`."""

    def run(self):
        """User-provided method that executes the task parameterised by the
        attributes of the task.

        Usually executed by [`Lab.run_tasks()`][labtech.Lab.run_tasks]
        instead of being called directly.

        """


def is_task_type(cls):
    """Returns `True` if the given `cls` is a class decorated with
    [`labtech.task`][labtech.task]."""
    return isclass(cls) and hasattr(cls, '_lt')


def is_task(obj):
    """Returns `True` if the given `obj` is an instance of a task class."""
    return hasattr(obj, '_is_task')


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
    def file_handle(self, key: str, filename: str, *, mode: str) -> IO:
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
    def save(self, storage: Storage, task: Task, result: Any) -> None:
        """Save the given `result` for the given `task` into the given
        `storage`."""

    @abstractmethod
    def load_task(self, storage: Storage, task_type: Type[Task], key: str) -> Task:
        """Loads the task instance of the given `task_type` for the given
        `key` from the given `storage`.

        The task's result is not loaded by this method.
        """

    @abstractmethod
    def load_result(self, storage: Storage, task: Task) -> Any:
        """Loads the result for the given task from the storage provider.

        Args:
            storage: Storage provider to load the result from
            task: task instance to load the result for

        """

    @abstractmethod
    def delete(self, storage: Storage, task: Task) -> None:
        """Deletes the cached result for the given `task` from the given
        `storage`."""
