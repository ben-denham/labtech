"""Cache classes that control formatting of task results in storage."""


from abc import abstractmethod
from datetime import datetime
import hashlib
import json
import pickle
from typing import Any, Optional, Type

from . import __version__ as labtech_version
from .types import Task, Cache, Storage
from .exceptions import CacheError, TaskNotFound
from .serialization import Serializer


class NullCache(Cache):
    """Cache that never stores results in the storage provider."""

    def cache_key(self, task: Task) -> str:
        return 'null'

    def is_cached(self, storage: Storage, task: Task) -> bool:
        return False

    def save(self, storage: Storage, task: Task, result: Any):
        pass

    def load_task(self, storage: Storage, task_type: Type[Task], key: str) -> Task:
        raise TaskNotFound

    def load_result(self, storage: Storage, task: Task) -> Any:
        raise CacheError('Loading a result from a NullCache is not supported.')

    def delete(self, storage: Storage, task: Task):
        pass


class BaseCache(Cache):
    """Base class for defining a Cache that will store results in a
    storage provider."""

    KEY_PREFIX = ''
    """Prefix for all files created by this Cache type - should be
    different for each Cache type to avoid conflicts."""

    METADATA_FILENAME = 'metadata.json'

    def __init__(self, *, serializer: Optional[Serializer] = None):
        self.serializer = serializer or Serializer()

    def cache_key(self, task: Task) -> str:
        serialized_str = json.dumps(self.serializer.serialize_task(task)).encode('utf-8')
        # Use sha1, as it is the same hash as git, produces short
        # hashes, and security concerns with sha1 are not relevant to
        # our use case.
        hashed = hashlib.sha1(serialized_str).hexdigest()
        return f'{self.KEY_PREFIX}{task.__class__.__qualname__}__{hashed}'

    def is_cached(self, storage: Storage, task: Task) -> bool:
        return storage.exists(task.cache_key)

    def save(self, storage: Storage, task: Task, result: Any):
        metadata = {
            'labtech_version': labtech_version,
            'cache': self.__class__.__qualname__,
            'cache_key': task.cache_key,
            'datetime': datetime.now().isoformat(),
            'task': self.serializer.serialize_task(task),
        }
        metadata_file = storage.file_handle(task.cache_key, self.METADATA_FILENAME, mode='w')
        with metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        self.save_result(storage, task, result)

    def load_task(self, storage: Storage, task_type: Type[Task], key: str) -> Task:
        if not key.startswith(f'{self.KEY_PREFIX}{task_type.__qualname__}'):
            raise TaskNotFound
        with storage.file_handle(key, self.METADATA_FILENAME, mode='r') as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get('cache') != self.__class__.__qualname__:
            raise TaskNotFound
        task = self.serializer.deserialize_task(metadata['task'])
        if not isinstance(task, task_type):
            raise TaskNotFound
        return task

    def delete(self, storage: Storage, task: Task):
        storage.delete(task.cache_key)

    @abstractmethod
    def save_result(self, storage: Storage, task: Task, result: Any):
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

    def __init__(self, *, serializer: Optional[Serializer] = None,
                 pickle_protocol: int = pickle.HIGHEST_PROTOCOL):
        super().__init__(serializer=serializer)
        self.pickle_protocol = pickle_protocol

    def save_result(self, storage: Storage, task: Task, result: Any):
        data_file = storage.file_handle(task.cache_key, self.RESULT_FILENAME, mode='wb')
        with data_file:
            pickle.dump(result, data_file, protocol=self.pickle_protocol)

    def load_result(self, storage: Storage, task: Task) -> Any:
        data_file = storage.file_handle(task.cache_key, self.RESULT_FILENAME, mode='rb')
        with data_file:
            return pickle.load(data_file)
