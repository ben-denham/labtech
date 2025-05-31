"""General labtech utilities."""
from __future__ import annotations

import builtins
import logging
import platform
import re
import sys
from multiprocessing import get_all_start_methods
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as base_tqdm_notebook

if TYPE_CHECKING:
    from collections.abc import Sequence


def make_logger_handler(*, task_name_placeholder: str = '%(processName)s') -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f'%(asctime)s/{task_name_placeholder}/%(levelname)s: %(message)s',
        '%Y-%m-%d %H:%M:%S',
    ))
    return handler


def get_logger():
    logger = logging.getLogger('labtech')
    logger.addHandler(make_logger_handler())
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
"""`logging.Logger` object that labtech logs events to during task execution.

Can be used to customize logging and to write additional logs from
task `run()` methods:

```python
import logging
from labtech import logger

# Change verbosity of logging
logger.setLevel(logging.ERROR)

# Logging methods to call from inside your task's run() method:
logger.info('Useful info from task: ...')
logger.warning('Warning from task: ...')
logger.error('Error from task: ...')
```

"""


T = TypeVar('T')


class OrderedSet(Generic[T]):
    """A set that returns items in the order they were added when
    iterated over."""

    def __init__(self, items: Sequence[T] | None = None):
        self.values: dict[T, T] = {}
        if items is not None:
            for item in items:
                self.add(item)

    def __str__(self):
        items_str = ', '.join([str(item) for item in self.values])
        return f'{{{items_str}}}'

    def __repr__(self):
        items_repr = ', '.join([repr(item) for item in self.values])
        return f'{{{items_repr}}}'

    def add(self, item: T):
        self.values[item] = item

    def remove(self, item: T):
        del self.values[item]

    def __contains__(self, item: T):
        return item in self.values

    def __iter__(self):
        return iter(self.values)

    def __add__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        combined: OrderedSet[T] = OrderedSet()
        combined.values.update(self.values)
        combined.values.update(other.values)
        return combined

    def __len__(self):
        return len(self.values)


class LoggerFileProxy:
    """File-like object that can replace sys.stdout and sys.stderr to
    redirect their streams to a logger_func with an added prefix."""

    whitespace_only_re = re.compile(r'[\s]*')

    def __init__(self, logger_func, prefix):
        self.logger_func = logger_func
        self.prefix = prefix
        self.bufs = []

    def write(self, buf):
        if not self.whitespace_only_re.fullmatch(buf):
            self.bufs.append(buf)

    def flush(self):
        if self.bufs:
            self.logger_func('\n'.join([f'{self.prefix}{buf}' for buf in self.bufs]))
            self.bufs = []


def ensure_dict_key_str(value, *, exception_type: type[Exception]) -> str:
    if not isinstance(value, str):
        raise exception_type(("Parameter dictionary keys must be strings, "
                              f"found: '{value}'"))
    return cast('str', value)


def is_interactive() -> bool:
    return hasattr(sys, 'ps1')


def is_ipython() -> bool:
    return hasattr(builtins, '__IPYTHON__')


def get_supported_start_methods() -> list[str]:
    start_methods = get_all_start_methods()
    # Even though macOS reports that it can support process forking,
    # it is not considered safe: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    if 'fork' in start_methods and platform.system() == 'Darwin':
        start_methods.remove('fork')
    return start_methods


# Disable tqdm monitoring, as we need to avoid threads if we'll be
# using fork (see: https://docs.python.org/3/library/os.html#os.fork)
class tqdm(base_tqdm):
    monitor_interval = 0


class tqdm_notebook(base_tqdm_notebook):
    monitor_interval = 0


__all__ = [
    'logger',
    'OrderedSet',
    'LoggerFileProxy',
    'ensure_dict_key_str',
    'is_interactive',
    'is_ipython',
    'tqdm',
    'tqdm_notebook',
]
