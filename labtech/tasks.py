"""Utilities for defining tasks."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from inspect import isclass
from typing import TYPE_CHECKING, cast

from frozendict import frozendict

from .cache import NullCache, PickleCache
from .exceptions import TaskError
from .types import TaskInfo, is_task, is_task_type
from .utils import ensure_dict_key_str

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import UnionType
    from typing import Any, TypeAlias

    from .types import Cache, LabContext, ResultMeta, ResultsMap, ResultT, Task

ParamScalar: TypeAlias = None | str | bool | float | int | Enum

class CacheDefault:
    pass


CACHE_DEFAULT = CacheDefault()


_RESERVED_ATTRS = [
    '_lt', '_is_task', '_cache_key', 'cache_key', 'result', '_results_map', '_set_results_map',
    'result_meta', '_set_result_meta', 'context', 'set_context', '__post_init__',
    '_set_code_version', 'code_version', 'current_code_version',
]
"""Reserved attribute names for task types."""


def immutable_param_value(key: str, value: Any) -> Any:
    """Converts a parameter value to an immutable equivalent that is hashable."""
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(immutable_param_value(f'{key}[{i}]', item) for i, item in enumerate(value))
    if isinstance(value, dict) or isinstance(value, frozendict):
        return frozendict({
            ensure_dict_key_str(dict_key, exception_type=TaskError): immutable_param_value(f'{key}["{dict_key}"]', dict_value)
            for dict_key, dict_value in value.items()
        })
    is_scalar = isinstance(value, cast('UnionType', ParamScalar))
    if is_scalar or is_task(value):
        return value
    raise TaskError(f"Unsupported type '{type(value).__qualname__}' in parameter value '{key}'.")


def _task_post_init(self: Task):
    # Ensure parameter values are immutable.
    for f in fields(self):
        object.__setattr__(self, f.name, immutable_param_value(f.name, getattr(self, f.name)))

    object.__setattr__(self, '_is_task', True)
    object.__setattr__(self, '_results_map', None)
    object.__setattr__(self, '_cache_key', None)
    object.__setattr__(self, 'code_version', self._lt.current_code_version)
    object.__setattr__(self, 'context', None)
    object.__setattr__(self, 'result_meta', None)
    if self._lt.orig_post_init is not None:
        self._lt.orig_post_init(self)


def _task_set_results_map(self: Task, results_map: ResultsMap):
    object.__setattr__(self, '_results_map', results_map)


def _task_set_result_meta(self: Task, result_meta: ResultMeta):
    object.__setattr__(self, 'result_meta', result_meta)


def _task_set_code_version(self: Task, code_version: str | None):
    # cache_key depends on the code_version, so clear any cached
    # cache_key:
    object.__setattr__(self, '_cache_key', None)
    object.__setattr__(self, 'code_version', code_version)


def _task_set_context(self: Task, context: LabContext):
    object.__setattr__(self, 'context', context)


def _task_filter_context_default(self: Task, context: LabContext) -> LabContext:
    return context


def _task_runner_options_default(self: Task) -> dict[str, Any]:
    return {}


def _task_current_code_version(self: Task) -> str | None:
    return self._lt.current_code_version


def _task_cache_key(self: Task) -> str:
    if self._cache_key is None:
        object.__setattr__(self, '_cache_key', self._lt.cache.cache_key(self))
    return cast('str', self._cache_key)


def _task_result(self: Task[ResultT]) -> ResultT:
    if hasattr(self, '_result'):
        return self._result
    if self._results_map is None:
        raise TaskError(f"Task '{self}' has no active results map")
    if self not in self._results_map:
        raise TaskError(f"Result for task '{self}' is not available in memory")
    return self._results_map[self].value


def _task__getstate__(self: Task) -> dict[str, Any]:
    state = {
        **{f.name: getattr(self, f.name) for f in fields(self)},
        '_lt': self._lt,
        '_is_task': self._is_task,
        'code_version': self.code_version,
        # Always pre-calculate the cache_key before pickling as _cache_key
        '_cache_key': self.cache_key,
        # We will never pickle the full _results_map or _result
        '_results_map': None,
    }
    return state


def _task__setstate__(self: Task, state: dict[str, Any]) -> None:
    field_set = set(f.name for f in fields(self))
    for key, value in state.items():
        value = immutable_param_value(key, value) if key in field_set else value
        object.__setattr__(self, key, value)


def task(*args,
         code_version: str | None = None,
         cache: CacheDefault | None | Cache = CACHE_DEFAULT,
         max_parallel: int | None = None,
         mlflow_run: bool = False):
    """Class decorator for defining task type classes.

    Task types are frozen [`dataclasses`], and attribute definitions
    should capture all parameters of the task type. Parameter
    attributes can be any of the following types:

    * Simple scalar types: `str`, `bool`, `float`, `int`, `None`
    * Any member of an `Enum` type. Referring to members of an `Enum` can be
      used to parameterise a task with a value that does not have one of the
      types above (e.g. a Pandas/Numpy dataset).
    * Task types: A task parameter is a "nested task" that will be executed
      before its parent so that it may make use of the nested result.
    * Collections of any of these types: `list`, `tuple`,
      `dict`, [`frozendict`](https://pypi.org/project/frozendict/)
      * Dictionaries may only contain string keys.
      * Note: Mutable `list` and `dict` collections will be converted to
        immutable `tuple` and [`frozendict`](https://pypi.org/project/frozendict/)
        collections.

    The task type is expected to define a `run()` method that takes no
    arguments (other than `self`). The `run()` method should execute
    the task as parameterised by the task's attributes and return the
    result of the task.

    Example usage:

    ```python
    @labtech.task
    class Experiment:
        seed: int
        multiplier: int

        def run(self):
            return self.seed * self.multiplier.value

    experiment = Experiment(seed=1, multiplier=2)
    ```

    You can also provide arguments to the decorator to control
    caching, cache-busting versioning, parallelism, and
    [mlflow](https://mlflow.org/docs/latest/tracking.html#quickstart)
    tracking:

    ```python
    @labtech.task(cache=None, code_version='v1', max_parallel=1, mlflow_run=True)
    class Experiment:
        ...

        def run(self):
            ...
    ```

    If a `post_init(self)` method is defined, it will be called after
    the task object is initialised (analagously to the `__post_init__`
    method of a dataclass). Because task types are frozen dataclasses,
    attributes can only be assigned to the task with
    `object.__setattr__(self, attribute_name, attribute_value)`.

    If a `filter_context(self, context: LabContext) -> LabContext`
    method is defined, it will be called to transform the context
    provided to each task. This can be useful for selecting subsets of
    large contexts in order to reduce data transferred to non-forked
    subprocesses or other kinds of processes in parallel processing
    frameworks. The filtering may take into account the values of the
    task's attributes. If `filter_context()` is not defined, the full
    context will be provided to each task.

    A `runner_options(self) -> dict[str, Any]` method may be defined
    to provide a dictionary of options to further control the
    behaviour of specific types of runner backend - refer to the
    documentation of each runner backend for supported options. The
    implementation may make use of the task's parameter values.

    Args:
        cache: The Cache that controls how task results are formatted for
            caching. Can be set to an instance of any
            [`Cache`](caching.md#caches) class, or `None` to disable caching
            of this type of task. Defaults to a
            [`PickleCache`][labtech.cache.PickleCache].
        code_version: Optional identifier for the version of task's
            implementation. Task results will only be loaded from the
            cache where they have a matching code_version, so you
            should change the code_version whenever the definition of
            the task's `run` method or any code it depends on changes
            in a way that will impact the result. Any string value can
            be used, e.g. 'v1', '2025-04-22', etc.
        max_parallel: The maximum number of instances of this task type that
            are allowed to run simultaneously in separate sub-processes. Useful
            to set if running too many instances of this particular task
            simultaneously will exhaust system memory or processing resources.
        mlflow_run: If True, the execution of each instance of this task type
            will be wrapped with `mlflow.start_run()`, tags the run with
            `labtech_task_type` equal to the task class name, and all parameters
            will be logged with `mlflow.log_param()`. You can make additional
            mlflow logging calls from the task's `run()` method.

    """

    def decorator(cls):
        nonlocal cache

        if not is_task_type(cls):
            for reserved_attr in _RESERVED_ATTRS:
                if hasattr(cls, reserved_attr):
                    raise AttributeError(f"Task type already defines reserved attribute '{reserved_attr}'.")

        post_init = getattr(cls, 'post_init', None)
        cls.__post_init__ = _task_post_init

        cls = dataclass(frozen=True, eq=True, order=True)(cls)

        run_func = getattr(cls, 'run', None)
        if not callable(run_func):
            raise NotImplementedError(f"Task type '{cls.__name__}' must define a 'run' method")

        if cache is CACHE_DEFAULT:
            cache = PickleCache()
        elif cache is None:
            cache = NullCache()

        cls._lt = TaskInfo(
            cache=cast('Cache', cache),
            orig_post_init=post_init,
            max_parallel=max_parallel,
            mlflow_run=mlflow_run,
            current_code_version=code_version,
        )
        cls.__getstate__ = _task__getstate__
        cls.__setstate__ = _task__setstate__
        cls._set_results_map = _task_set_results_map
        cls._set_result_meta = _task_set_result_meta
        cls._set_code_version = _task_set_code_version
        cls.current_code_version = property(_task_current_code_version)
        cls.cache_key = property(_task_cache_key)
        cls.result = property(_task_result)
        cls.set_context = _task_set_context
        if not hasattr(cls, 'filter_context'):
            cls.filter_context = _task_filter_context_default
        if not hasattr(cls, 'runner_options'):
            cls.runner_options = _task_runner_options_default
        return cls

    if len(args) > 0 and isclass(args[0]):
        return decorator(args[0], *args[1:])
    else:
        return decorator


def find_tasks_in_param(param_value: Any, searched_coll_ids: set[int] | None = None) -> Sequence[Task]:
    """Given a parameter value, return all tasks within it found through a recursive search."""
    if searched_coll_ids is None:
        searched_coll_ids = set()
    if id(param_value) in searched_coll_ids:
        return []

    if is_task(param_value):
        return [param_value]
    elif isinstance(param_value, list) or isinstance(param_value, tuple):
        searched_coll_ids = searched_coll_ids | {id(param_value)}
        return [
            task
            for item in param_value
            for task in find_tasks_in_param(item, searched_coll_ids)
        ]
    elif isinstance(param_value, dict) or isinstance(param_value, frozendict):
        searched_coll_ids = searched_coll_ids | {id(param_value)}
        return [
            task
            # We only need to search the values, as all parameter
            # dictionary keys must be strings.
            for item in param_value.values()
            for task in find_tasks_in_param(item, searched_coll_ids)
        ]
    elif isinstance(param_value, cast('UnionType', ParamScalar)):
        return []

    # This should be impossible.
    msg = f"Unexpected type {type(param_value).__qualname__} encountered in task parameter value."
    raise TaskError(msg)


def get_direct_dependencies(task: Task, *, all_identities: bool) -> list[Task]:
    """Return a list of tasks that are direct (first-level)
    dependencies of the given task in its attributes.

    If all_identities=True, then all duplicate identities of the same
    task will be included in the output.

    """
    identifier_to_dependency_task: dict[Task | int, Task] = {}

    for field in fields(task):
        field_value = getattr(task, field.name)
        for dependency_task in find_tasks_in_param(field_value):
            identifier = id(dependency_task) if all_identities else dependency_task
            identifier_to_dependency_task[identifier] = dependency_task
    return list(identifier_to_dependency_task.values())
