"""Storage providers for cached task results."""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING

from .exceptions import StorageError
from .types import Storage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import IO


def validate_file_path_key(key: str, *, storage_path: Path) -> None:
    if not key:
        raise StorageError("Key cannot be empty")

    disallowed_key_chars = ['.', '/', '\\', os.path.sep, os.path.altsep]
    for char in disallowed_key_chars:
        if char is not None and char in key: # altsep can be None
            raise StorageError(f"Key '{key}' must not contain the forbidden character '{char}'")

    key_path = (storage_path / key).resolve()
    if key_path.parent != storage_path.resolve():
        raise StorageError((f"Key '{key}' should only reference a directory directly "
                            f"under the storage directory '{storage_path}'"))


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

    def __init__(self, storage_dir: str | Path, *,
                 runner_dir: str | Path | None = None,
                 with_gitignore: bool = True):
        """
        Args:
            storage_dir: Path to the directory where cached results will be
                stored. The directory will be created if it does not already
                exist.
            runner_dir: Alternative directory path to be used from within runner
                tasks (e.g. if runner tasks will run on a separate machine with
                different paths).
            with_gitignore: If `True`, a `.gitignore` file will be created
                inside the storage directory to ignore the entire storage
                directory. If an existing `.gitignore` file exists, it will be
                replaced.
        """
        self._runner_dir = runner_dir

        if isinstance(storage_dir, str):
            storage_dir = Path(storage_dir)
        self._storage_path = storage_dir.resolve()
        if not self._storage_path.exists():
            self._storage_path.mkdir()

        self._with_gitignore = with_gitignore
        if with_gitignore:
            gitignore_path = self._storage_path / '.gitignore'
            with gitignore_path.open('w') as gitignore_file:
                gitignore_file.write('*\n')

    def _key_to_path(self, key: str) -> Path:
        validate_file_path_key(key, storage_path=self._storage_path)
        return (self._storage_path / key).resolve()

    def find_keys(self) -> Sequence[str]:
        return sorted([
            key_path.name for key_path in self._storage_path.iterdir()
            if key_path.is_dir()
        ])

    def exists(self, key: str) -> bool:
        key_path = self._key_to_path(key)
        return key_path.exists()

    def file_handle(self, key: str, filename: str, *, mode: str = 'r') -> IO:
        key_path = self._key_to_path(key)
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
        key_path = self._key_to_path(key)
        if key_path.exists():
            shutil.rmtree(key_path)

    def get_runner_storage(self) -> LocalStorage:
        if self._runner_dir is None:
            return self
        return LocalStorage(
            self._runner_dir,
            with_gitignore=self._with_gitignore,
        )


class FsspecStorage(Storage, ABC):
    """Base class for using an
    [`fsspec`](https://filesystem-spec.readthedocs.io) filesystem for
    storage."""

    def __init__(self, storage_dir: str | Path):
        """
        Args:
            storage_dir: Path to the directory where cached results will be
                stored.
        """
        if isinstance(storage_dir, str):
            # Default to posix paths, which should work for most
            # fsspec filesystems.
            storage_dir = PosixPath(storage_dir)
        self._storage_path = storage_dir
        fs = self.fs_constructor()
        fs.mkdirs(str(self._storage_path), exist_ok=True)

    def _key_to_path(self, key):
        validate_file_path_key(key, storage_path=self._storage_path)
        return self._storage_path / key

    def find_keys(self) -> Sequence[str]:
        fs = self.fs_constructor()
        return [
            str(Path(entry).relative_to(self._storage_path))
            for entry in fs.ls(str(self._storage_path))
        ]

    def exists(self, key: str) -> bool:
        fs = self.fs_constructor()
        return fs.exists(str(self._key_to_path(key)))

    def file_handle(self, key: str, filename: str, *, mode: str = 'r') -> IO:
        fs = self.fs_constructor()
        key_path = self._key_to_path(key)
        fs.mkdirs(str(key_path), exist_ok=True)
        file_path = key_path / filename
        if file_path.parent != key_path:
            raise ValueError((f"Filename '{filename}' should only reference a directory directly "
                              f"under the storage key directory '{key_path}'"))
        return fs.open(str(file_path), mode)

    def delete(self, key: str):
        fs = self.fs_constructor()
        path = self._key_to_path(key)
        if fs.exists(str(path)):
            fs.rm(str(path), recursive=True)

    @abstractmethod
    def fs_constructor(self):
        """Return an [`fsspec.AbstractFileSystem`](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem)
        to use for file storage."""
        pass


# Useful reference implementations:

# from s3fs.implementations import LocalFileSystem
# class LocalFsspecStorage(FsspecStorage):
#     def __init__(self, storage_dir: Union[str, Path], **kwargs):
#         if isinstance(storage_dir, str):
#             # Use system-local Path type.
#             storage_dir = Path(storage_dir)
#         super().__init__(storage_dir.resolve())
#     def fs_constructor(self):
#         return LocalFileSystem()

# from s3fs import S3FileSystem
# class S3fsStorage(FsspecStorage):
#     def fs_constructor(self):
#         return S3FileSystem(
#             # Use localstack endpoint:
#             endpoint_url='http://localhost:4566',
#             key='anything',
#             secret='anything',
#         )
