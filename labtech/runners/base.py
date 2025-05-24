from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from frozendict import frozendict

from labtech.exceptions import LabError
from labtech.tasks import is_task
from labtech.types import ResultMeta, TaskResult
from labtech.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from labtech.types import LabContext, Storage, Task


@contextmanager
def optional_mlflow(task: Task):
    """Context manager to set mlflow "run" configuration for a task if
    mlflow_run=True is set for the task type."""

    def log_params(value: Any, *, path: str = ''):
        prefix = path if path == '' else f'{path}.'
        if is_task(value):
            for field in fields(value):
                log_params(getattr(value, field.name), path=f'{prefix}{field.name}')
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                log_params(item, path=f'{prefix}{i}')
        elif isinstance(value, frozendict):
            for key, item in value.items():
                log_params(item, path=f'{prefix}{key}')
        elif isinstance(value, Enum):
            mlflow.log_param(path, f'{type(value).__qualname__}.{value.name}')
        elif ((value is None)
              or isinstance(value, str)
              or isinstance(value, bool)
              or isinstance(value, float)
              or isinstance(value, int)):
            mlflow.log_param(path, value)
        else:
            raise LabError(
                (f"Unable to mlflow log parameter '{path}' of type '{type(value).__qualname__}' "
                 f"in task of type '{type(task).__qualname__}'.")
            )

    if task._lt.mlflow_run:
        try:
            import mlflow
        except ImportError:
            raise LabError(
                (f"Task type '{type(task).__qualname__}' is configured with mlflow_run=True, but "
                 "mlflow cannot be imported. You can install mlflow with `pip install mlflow`.")
            )
        with mlflow.start_run():
            mlflow.set_tag('labtech_task_type', type(task).__qualname__)
            log_params(task)
            yield
    else:
        yield


def run_or_load_task(task: Task, use_cache: bool, filtered_context: LabContext, storage: Storage) -> TaskResult:
    """Called by a Runner to either:

    1. Run the task with the given filtered_context and cache its result in the given
       storage and return the result OR
    2. Load and return the task's result from the cache if it is present
       and use_cache=True

    """
    if use_cache:
        logger.debug(f"Loading from cache: '{task}'")
        task_result = task._lt.cache.load_result_with_meta(storage, task)
        return task_result
    else:
        logger.debug(f"Running: '{task}'")
        task.set_context(filtered_context)
        with optional_mlflow(task):
            start = datetime.now()
            result = task.run()
            end = datetime.now()
        task_result = TaskResult(
            value=result,
            meta=ResultMeta(
                start=start,
                duration=(end - start),
            ),
        )
        task._lt.cache.save(storage, task, task_result)
        logger.debug(f"Completed: '{task}'")
        return task_result
