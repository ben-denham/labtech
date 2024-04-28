"""Storage providers for cached task results."""

import os
import shutil
from pathlib import Path
from typing import IO, Sequence, Union

from .exceptions import StorageError
from .types import Storage


class NullStorage(Storage):
    """Storage provider that does not store cached results."""

    def find_keys(self) -> Sequence[str]:
        return []

    def exists(self, key: str) -> bool:
        return False

    def file_handle(self, key: str, filename: str, *, mode: str = 'r') -> IO:
        return open(os.devnull, mode=mode)

    def delete(self, key: str):
        pass


class LocalStorage(Storage):
    """Storage provider that stores cached results in a local filesystem
    directory."""

    def __init__(self, storage_dir: Union[str, Path], *, with_gitignore: bool = True):
        """
        Args:
            storage_dir: Path to the directory where cached results will be
                stored. The directory will be created if it does not already
                exist.
            with_gitignore: If `True`, a `.gitignore` file will be created
                inside the storage directory to ignore the entire storage
                directory. If an existing `.gitignore` file exists, it will be
                replaced.
        """
        if isinstance(storage_dir, str):
            storage_dir = Path(storage_dir)
        self._storage_path = storage_dir.resolve()
        if not self._storage_path.exists():
            self._storage_path.mkdir()

        if with_gitignore:
            gitignore_path = self._storage_path / '.gitignore'
            with gitignore_path.open('w') as gitignore_file:
                gitignore_file.write('*\n')

    def _key_path(self, key: str) -> Path:
        if not key:
            raise StorageError("Key cannot be empty")

        if key.startswith('/') or key.endswith('/') or key.startswith(os.path.sep) or key.endswith(os.path.sep):
            msg = f"Key '{key}' should not start or end with '/'"
            if os.path.sep != '/':
                msg += f" or '{os.path.sep}'"
            raise StorageError(msg)

        key_path = (self._storage_path / key).resolve()
        if key_path.parent != self._storage_path:
            raise StorageError((f"Key '{key}' should only reference a directory directly "
                                f"under the storage directory '{self._storage_path}'"))
        return key_path

    def find_keys(self) -> Sequence[str]:
        return sorted([
            key_path.name for key_path in self._storage_path.iterdir()
            if key_path.is_dir()
        ])

    def exists(self, key: str) -> bool:
        key_path = self._key_path(key)
        return key_path.exists()

    def file_handle(self, key: str, filename: str, *, mode: str = 'r') -> IO:
        key_path = self._key_path(key)
        try:
            key_path.mkdir()
        except FileExistsError:
            pass
        file_path = (key_path / filename).resolve()
        if file_path.parent != key_path:
            raise StorageError((f"Filename '{filename}' should only reference a directory directly "
                                f"under the storage key directory '{key_path}'"))
        return file_path.open(mode=mode)

    def delete(self, key: str):
        key_path = self._key_path(key)
        if key_path.exists():
            shutil.rmtree(key_path)
