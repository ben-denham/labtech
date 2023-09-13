"""Utilities for defining tasks."""

from dataclasses import dataclass, fields
from inspect import isclass
from typing import cast, Any, Dict, Optional, Union

from .types import TaskInfo, ResultsMap, Cache, is_task_type
from .cache import PickleCache, NullCache
from .exceptions import TaskError


class CacheDefault:
    pass


CACHE_DEFAULT = CacheDefault()


def _task_post_init(self):
    object.__setattr__(self, '_is_task', True)
    object.__setattr__(self, 'cache_key', self._lt.cache.cache_key(self))
    object.__setattr__(self, '_results_map', None)
    if self._lt.orig_post_init is not None:
        self._lt.orig_post_init(self)


def _task_set_results_map(self, results_map: ResultsMap):
    object.__setattr__(self, '_results_map', results_map)


def _task_result(self) -> Any:
    if hasattr(self, '_result'):
        return self._result
    if self._results_map is None:
        raise TaskError(f"Task '{self}' has no active results map")
    if self not in self._results_map:
        raise TaskError(f"Result for task '{self}' is not available in memory")
    return self._results_map[self]


def _task__getstate__(self) -> Dict[str, Any]:
    state = {
        **{f.name: getattr(self, f.name) for f in fields(self)},
        '_lt': self._lt,
        '_is_task': self._is_task,
        'cache_key': self.cache_key,
        # We will never pickle the full _results_map
        '_results_map': None,
    }
    if hasattr(self, '_result'):
        state['_result'] = self._result
    elif self._results_map is not None:
        # Avoid pickling the whole _results_map, and also avoid
        # AttributeError when attempting to pickle self inside object.
        state['_result'] = self._results_map.get(self)
    return state


def _task__setstate__(self, state: Dict[str, Any]) -> None:
    for key, value in state.items():
        object.__setattr__(self, key, value)


def task(*args,
         cache: Union[CacheDefault, None, Cache] = CACHE_DEFAULT,
         max_parallel: Optional[int] = None):
    """Class decorator for defining task type classes.

    Attribute definitions in task types are handled in the same way as
    [`dataclasses`], and they should capture all parameters of the
    task type. Parameter attributes can be any of the following types:

    * Simple scalar types: `str`, `bool`, `float`, `int`, `None`
    * Collections of any of these types: `list`, `tuple`, `dict`, `Enum`
    * Task types: A task parameter is a "nested task" that will be executed
      before its parent so that it may make use of the nested result.

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

    You can also provide arguments to the decorator to control caching
    and parallelism:

    ```python
    @labtech.task(cache=None, max_parallel=1)
    class Experiment:
        ...

        def run(self):
            ...
    ```

    Args:
        cache: The Cache that controls how task results are formatted for
            caching. Can be set to an instance of any
            [`Cache`](caching.md#caches) class, or `None` to disable caching
            of this type of task. Defaults to a
            [`PickleCache`][labtech.cache.PickleCache].
        max_parallel: The maximum number of instances of this task type that
            are allowed to run simultaneously in separate sub-processes. Useful
            to set if running too many instances of this particular task
            simultaneously will exhaust system memory or processing resources.

    """

    def decorator(cls):
        nonlocal cache

        reserved_attrs = ['_lt', '_is_task', 'cache_key', 'result', '_results_map', '_set_results_map']
        if not is_task_type(cls):
            for reserved_attr in reserved_attrs:
                if hasattr(cls, reserved_attr):
                    raise AttributeError(f"Task type already defines reserved attribute '{reserved_attr}'.")

        orig_post_init = getattr(cls, '__post_init__', None)
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
            cache=cast(Cache, cache),
            orig_post_init=orig_post_init,
            max_parallel=max_parallel,
        )
        cls.__getstate__ = _task__getstate__
        cls.__setstate__ = _task__setstate__
        cls._set_results_map = _task_set_results_map
        cls.result = property(_task_result)
        return cls

    if len(args) > 0 and isclass(args[0]):
        return decorator(args[0], *args[1:])
    else:
        return decorator
