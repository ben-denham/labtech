from inspect import isclass
from typing import TypedDict

from .exceptions import ParamHandlerError
from .types import ParamHandler


class ParamHandlerEntry(TypedDict):
    handler: ParamHandler
    priority: int


CUSTOM_PARAM_HANDLER_ENTRIES: dict[str, ParamHandlerEntry] = {}
CUSTOM_PARAM_HANDLERS: list[ParamHandler] = []


def param_handler(*args, priority: int = 1000):
    """Class decorator for declaring custom parameter handlers that
    can define how Labtech should handle the processing,
    serialization, and deserialization of additional parameter types.

    Defining a custom parameter handler is an advanced feature of
    Labtech, and you are responsible for ensuring:

    * The decorated class implements all methods of the
      [`ParamHandler`][labtech.types.ParamHandler] protocol.
    * To ensure tasks are reproducible, you should only define
      handlers for customer parameter types that are **immutable**.
    * Because tasks are hashable representations of their parameters,
      you should only define handlers for customer parameter types that
      are **hashable**.
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
        global CUSTOM_PARAM_HANDLERS

        if not isinstance(cls, ParamHandler):
            raise ParamHandlerError(
                (f"Cannot register '{cls.__qualname__}' as a custom parameter handler, "
                 "as it does not implement all methods of the 'ParamHandler' protocol.")
            )

        CUSTOM_PARAM_HANDLER_ENTRIES[f'{cls.__module__}.{cls.__qualname__}'] = ParamHandlerEntry(
            handler=cls(),
            priority=priority,
        )
        CUSTOM_PARAM_HANDLERS = [
            entry['handler'] for entry in
            # Sort param handlers by priority, keeping insertion order
            # where priorities are equal.
            sorted(CUSTOM_PARAM_HANDLERS.values(), key=lambda entry: entry['priority'])
        ]

        return cls

    if len(args) > 0 and isclass(args[0]):
        return decorator(args[0], *args[1:])
    else:
        return decorator
