import os
from pathlib import Path

import pytest

from labtech.exceptions import StorageError
from labtech.storage import LocalStorage


@pytest.fixture()
def local_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path)

class TestLocalStorage:
    class TestDunderInit:
        def test_str(self, tmp_path: Path):
            tmp_path_str = tmp_path.as_posix()

            storage = LocalStorage(tmp_path_str)
            assert isinstance(storage._storage_path, Path)
            assert storage._storage_path.exists()
            assert storage._storage_path == tmp_path

        def test_path(self, tmp_path: Path):
            storage = LocalStorage(tmp_path)
            assert isinstance(storage._storage_path, Path)
            assert storage._storage_path.exists()
            assert storage._storage_path == tmp_path

    class TestFindKeys:
        def test_empty(self, local_storage: LocalStorage):
            assert local_storage.find_keys() == []

        def test_single_key(self, local_storage: LocalStorage):
            key = 'key'
            (local_storage._storage_path / key).mkdir(exist_ok=True)
            assert local_storage.find_keys() == [key]

        def test_multiple_keys_sorted(self, local_storage: LocalStorage):
            keys = ['b', 'a', 'c']
            for key in keys:
                (local_storage._storage_path / key).mkdir(exist_ok=True)
            assert local_storage.find_keys() == sorted(keys)

    class TestExists:
        def test_missing_key(self, local_storage: LocalStorage):
            assert not local_storage.exists('fake')

        def test_existing_key(self, local_storage: LocalStorage):
            key = 'key'
            (local_storage._storage_path / key).touch()
            assert local_storage.exists(key)

        def test_empty_key_raises(self, local_storage: LocalStorage):
            with pytest.raises(StorageError):
                local_storage.exists('')

        def test_nested_key_raises(self, local_storage: LocalStorage):
            with pytest.raises(StorageError):
                local_storage.exists('nested/key')

        def test_key_with_slash_raises(self, local_storage: LocalStorage):
            with pytest.raises(StorageError):
                local_storage.exists('key/')

            with pytest.raises(StorageError):
                local_storage.exists('key' + os.path.sep)

            with pytest.raises(StorageError):
                local_storage.exists('/key')

            with pytest.raises(StorageError):
                local_storage.exists(os.path.sep + 'key')

        def test_backslash(self, local_storage: LocalStorage):
            with pytest.raises(StorageError):
                local_storage.exists('key\\with\\backslashes')

        def test_dot(self, local_storage: LocalStorage):
            with pytest.raises(StorageError):
                local_storage.exists('key.with.dots')

        def test_forwardslash(self, local_storage: LocalStorage):
            with pytest.raises(StorageError):
                local_storage.exists('key/with/forwardslashes')

        def test_relative_path_trickery(self, local_storage: LocalStorage):
            key = 'other_key'
            (local_storage._storage_path / key).touch()
            with pytest.raises(StorageError):
                local_storage.exists('key/../other_key')
