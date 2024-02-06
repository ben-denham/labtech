"""General labtech utilities."""

import logging
import re
from typing import cast, Dict, Generic, Optional, Sequence, TypeVar, Type


def get_logger():
    logger = logging.getLogger('labtech')
    default_logger_handler = logging.StreamHandler()
    logger.addHandler(default_logger_handler)
    default_logger_handler.setFormatter(logging.Formatter(
        '%(asctime)s/%(processName)s/%(levelname)s: %(message)s',
        '%Y-%m-%d %H:%M:%S',
    ))
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

    def __init__(self, items: Optional[Sequence[T]] = None):
        self.values: Dict[T, T] = {}
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

    def __add__(self, other: 'OrderedSet[T]') -> 'OrderedSet[T]':
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


def ensure_dict_key_str(value, *, exception_type: Type[Exception]) -> str:
    if not isinstance(value, str):
        raise exception_type(("Parameter dictionary keys must be strings, "
                              f"found: '{value}'"))
    return cast(str, value)


__all__ = [
    'logger',
    'OrderedSet',
    'LoggerFileProxy',
    'ensure_dict_key_str',
]
