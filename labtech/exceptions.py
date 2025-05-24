"""Custom exceptions that may be raised by labtech."""
from __future__ import annotations


class LabtechError(Exception):
    """Base for all exceptions raised by labtech."""


class LabError(LabtechError):
    """Raised for failures when interacting with Lab objects."""


class RunnerError(LabtechError):
    """Raised for failures when running Tasks."""


class TaskDiedError(RunnerError):
    """Raised when the execution of a task died unexpectedly."""

    def __init__(self):
        super().__init__('Task died')


class TaskError(LabtechError):
    """Raised for failures when handling Task objects."""


class StorageError(LabtechError):
    """Raised for failures when interacting with Storage objects."""


class SerializationError(LabtechError):
    """Raised for serialization and deserialization failures."""


class CacheError(LabtechError):
    """Raised for failures when interacting with Cache objects."""


class TaskNotFound(CacheError):
    """Raised when a task is not found in a Cache."""
