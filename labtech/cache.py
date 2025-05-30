"""Cache classes that control formatting of task results in storage."""
from __future__ import annotations

import hashlib
import json
import pickle
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from . import __version__ as labtech_version
from .exceptions import CacheError, TaskNotFound
from .serialization import Serializer
from .types import Cache, ResultMeta, TaskResult

if TYPE_CHECKING:
    from typing import Any

    from .types import ResultT, Storage, Task, TaskT


def _format_class_name(cls):
    """Format a qualified class name in format that can be safely
    represented in a filename."""
    # Nested classes may have periods that need to be replaced.
    return cls.__qualname__.replace('.', '-')


class NullCache(Cache):
    """Cache that never stores results in the storage provider."""

    def cache_key(self, task: Task) -> str:
        return 'null'

    def is_cached(self, storage: Storage, task: Task) -> bool:
        return False

    def save(self, storage: Storage, task: Task[ResultT], result: TaskResult[ResultT]):
        pass

    def load_task(self, storage: Storage, task_type: type[TaskT], key: str) -> TaskT:
        raise TaskNotFound

    def load_result_with_meta(self, storage: Storage, task: Task[ResultT]) -> TaskResult[ResultT]:
        raise CacheError('Loading a result from a NullCache is not supported.')

    def load_cache_timestamp(self, storage: Storage, task: Task) -> Any:
        raise CacheError('Loading a cache_timestamp from a NullCache is not supported.')

    def delete(self, storage: Storage, task: Task):
        pass


class BaseCache(Cache):
    """Base class for defining a Cache that will store results in a
    storage provider."""

    KEY_PREFIX = ''
    """Prefix for all files created by this Cache type - should be
    different for each Cache type to avoid conflicts."""

    METADATA_FILENAME = 'metadata.json'

    def __init__(self, *, serializer: Serializer | None = None):
        self.serializer = serializer or Serializer()

    def cache_key(self, task: Task) -> str:
        serialized_str = json.dumps(self.serializer.serialize_task(task)).encode('utf-8')
        # Use sha1, as it is the same hash as git, produces short
        # hashes, and security concerns with sha1 are not relevant to
        # our use case.
        hashed = hashlib.sha1(serialized_str).hexdigest()
        return f'{self.KEY_PREFIX}{_format_class_name(task.__class__)}__{hashed}'

    def is_cached(self, storage: Storage, task: Task) -> bool:
        return storage.exists(task.cache_key)

    def save(self, storage: Storage, task: Task[ResultT], task_result: TaskResult[ResultT]):
        start_timestamp = None
        if task_result.meta.start is not None:
            start_timestamp = task_result.meta.start.isoformat()

        duration_seconds = None
        if task_result.meta.duration is not None:
            duration_seconds = task_result.meta.duration.total_seconds()

        metadata = {
            'labtech_version': labtech_version,
            'cache': self.__class__.__qualname__,
            'cache_key': task.cache_key,
            'task': self.serializer.serialize_task(task),
            'start_timestamp': start_timestamp,
            'duration_seconds': duration_seconds,
        }
        metadata_file = storage.file_handle(task.cache_key, self.METADATA_FILENAME, mode='w')
        with metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        self.save_result(storage, task, task_result.value)

    def load_metadata(self, storage: Storage, task_type: type[Task], key: str) -> dict[str, Any]:
        if not key.startswith(f'{self.KEY_PREFIX}{_format_class_name(task_type)}'):
            raise TaskNotFound
        with storage.file_handle(key, self.METADATA_FILENAME, mode='r') as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get('cache') != self.__class__.__qualname__:
            raise TaskNotFound
        return metadata

    def build_result_meta(self, metadata: dict[str, Any]) -> ResultMeta:
        start = None
        if 'start_timestamp' in metadata:
            start = datetime.fromisoformat(metadata['start_timestamp'])

        duration = None
        if 'duration_seconds' in metadata:
            duration = timedelta(seconds=metadata['duration_seconds'])

        return ResultMeta(
            start=start,
            duration=duration,
        )

    def load_task(self, storage: Storage, task_type: type[TaskT], key: str) -> TaskT:
        metadata = self.load_metadata(storage, task_type, key)
        result_meta = self.build_result_meta(metadata)
        task = self.serializer.deserialize_task(metadata['task'], result_meta=result_meta)
        if not isinstance(task, task_type):
            raise TaskNotFound
        return task

    def load_result_with_meta(self, storage: Storage, task: Task[ResultT]) -> TaskResult[ResultT]:
        result = self.load_result(storage, task)
        metadata = self.load_metadata(storage, type(task), task.cache_key)
        return TaskResult(
            value=result,
            meta=self.build_result_meta(metadata),
        )

    def delete(self, storage: Storage, task: Task):
        storage.delete(task.cache_key)

    @abstractmethod
    def load_result(self, storage: Storage, task: Task[ResultT]) -> ResultT:
        """Loads the result for the given task from the storage provider.

        Args:
            storage: Storage provider to load the result from
            task: task instance to load the result for

        """

    @abstractmethod
    def save_result(self, storage: Storage, task: Task[ResultT], result: ResultT):
        """Saves the given task result into the storage provider.

        Args:
            storage: Storage provider to save the result into
            task: task instance the result belongs to
            result: result to save

        """


class PickleCache(BaseCache):
    """Default cache that stores results as
    [pickled](https://docs.python.org/3/library/pickle.html) Python
    objects.

    NOTE: As pickle is not secure, you should only load pickle cache
    results that you trust.

    """

    KEY_PREFIX = 'pickle__'
    RESULT_FILENAME = 'data.pickle'

    def __init__(self, *, serializer: Serializer | None = None,
                 pickle_protocol: int = pickle.HIGHEST_PROTOCOL):
        super().__init__(serializer=serializer)
        self.pickle_protocol = pickle_protocol

    def save_result(self, storage: Storage, task: Task[ResultT], result: ResultT):
        data_file = storage.file_handle(task.cache_key, self.RESULT_FILENAME, mode='wb')
        with data_file:
            pickle.dump(result, data_file, protocol=self.pickle_protocol)

    def load_result(self, storage: Storage, task: Task[ResultT]) -> ResultT:
        data_file = storage.file_handle(task.cache_key, self.RESULT_FILENAME, mode='rb')
        with data_file:
            return pickle.load(data_file)
