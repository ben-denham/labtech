#from functools import cached_property
from inspect import isclass
from typing import Optional, Type, TypedDict

from .exceptions import ParamHandlerError, UnregisteredParamHandlerError
from .types import ParamHandler
from .utils import fully_qualified_class_name


class ParamHandlerEntry(TypedDict):
    handler: ParamHandler
    priority: int


class ParamHandlerManager:

    def __init__(self) -> None:
        self._entries: dict[str, ParamHandlerEntry] = {}
        self._prioritised_handlers: Optional[list[ParamHandler]] = None

    def register(self, cls: Type[ParamHandler], *, priority: int) -> None:
        if not isinstance(cls, ParamHandler):
            raise ParamHandlerError(
                (f"Cannot register '{cls.__qualname__}' as a custom parameter handler, "
                 "as it does not implement all methods of the 'ParamHandler' protocol.")
            )

        self._entries[fully_qualified_class_name(cls)] = ParamHandlerEntry(
            handler=cls(),
            priority=priority,
        )
        # Clear cache
        self._prioritised_handlers = None

    def lookup(self, fq_class_name: str) -> ParamHandler:
        try:
            entry = self._entries[fq_class_name]
        except KeyError:
            raise UnregisteredParamHandlerError(fully_qualified_class_name)
        return entry['handler']

    def clear(self) -> None:
        self._entries = {}
        # Clear cache
        self._prioritised_handlers = None

    @property
    def prioritised_handlers(self) -> list[ParamHandler]:
        if self._prioritised_handlers is None:
            self._prioritised_handlers = [
                entry['handler'] for entry in
                # Sort param handlers by priority, keeping insertion order
                # where priorities are equal.
                sorted(self._entries.values(), key=lambda entry: entry['priority'])
            ]
        return self._prioritised_handlers

    def instantiate(self) -> None:
        global _PARAM_HANDLER_MANAGER
        _PARAM_HANDLER_MANAGER = self

    @staticmethod
    def get() -> 'ParamHandlerManager':
        return _PARAM_HANDLER_MANAGER


_PARAM_HANDLER_MANAGER = ParamHandlerManager()


def param_handler(*args, priority: int = 1000):
    """Class decorator for declaring custom parameter handlers that
    can define how Labtech should handle the processing,
    serialization, and deserialization of additional parameter types.

    Defining a custom parameter handler is an advanced feature of
    Labtech, and you are responsible for ensuring:

    * The decorated class implements all methods of the
      [`ParamHandler`][labtech.types.ParamHandler] protocol.
    * To ensure tasks are reproducible, you should only define
      handlers for custom parameter types that are **immutable and
      composed only of immutable elements**.
    * Because tasks are hashable representations of their parameters,
      you should only define handlers for custom parameter types that
      are **hashable and composed only of hashable elements**.
    * Because serialized parameters will reference the module path and
      class name of the custom parameter handler that was used to
      serialize them, you should avoid moving or renaming custom
      parameter handlers once they are in use.

    Args:
        priority: Determines the order in which custom parameter handlers are
            applied when processing a parameter value. Lower priority values
            are applied first.

    """

    def decorator(cls):
        ParamHandlerManager.get().register(cls, priority=priority)
        return cls

    if len(args) > 0 and isclass(args[0]):
        return decorator(args[0], *args[1:])
    else:
        return decorator
