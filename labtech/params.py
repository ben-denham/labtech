from inspect import isclass
from typing import TypedDict

from .exceptions import ParamHandlerError, UnregisteredParamHandlerError
from .types import ParamHandler
from .utils import fully_qualified_class_name


class ParamHandlerEntry(TypedDict):
    handler: ParamHandler
    priority: int


_CUSTOM_PARAM_HANDLER_ENTRIES: dict[str, ParamHandlerEntry] = {}
_CUSTOM_PARAM_HANDLERS = []


def _update_custom_param_handlers() -> None:
    global _CUSTOM_PARAM_HANDLERS
    _CUSTOM_PARAM_HANDLERS = [
        entry['handler'] for entry in
        # Sort param handlers by priority, keeping insertion order
        # where priorities are equal.
        sorted(_CUSTOM_PARAM_HANDLER_ENTRIES.values(), key=lambda entry: entry['priority'])
    ]


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
        global _CUSTOM_PARAM_HANDLERS

        if not isinstance(cls, ParamHandler):
            raise ParamHandlerError(
                (f"Cannot register '{cls.__qualname__}' as a custom parameter handler, "
                 "as it does not implement all methods of the 'ParamHandler' protocol.")
            )

        _CUSTOM_PARAM_HANDLER_ENTRIES[fully_qualified_class_name(cls)] = ParamHandlerEntry(
            handler=cls(),
            priority=priority,
        )
        _update_custom_param_handlers()

        return cls

    if len(args) > 0 and isclass(args[0]):
        return decorator(args[0], *args[1:])
    else:
        return decorator


def get_custom_param_handler_entries() -> dict[str, ParamHandlerEntry]:
    return _CUSTOM_PARAM_HANDLER_ENTRIES


def set_custom_param_handler_entries(custom_param_handler_entries: dict[str, ParamHandlerEntry]) -> None:
    global _CUSTOM_PARAM_HANDLER_ENTRIES
    _CUSTOM_PARAM_HANDLER_ENTRIES = custom_param_handler_entries
    _update_custom_param_handlers()


def get_custom_param_handlers() -> list[ParamHandler]:
    return _CUSTOM_PARAM_HANDLERS


def lookup_custom_param_handler(fq_class_name: str) -> ParamHandler:
    try:
        entry = _CUSTOM_PARAM_HANDLER_ENTRIES[fq_class_name]
    except KeyError:
        raise UnregisteredParamHandlerError(fully_qualified_class_name)
    return entry['handler']


def clear_custom_param_handlers() -> None:
    global _CUSTOM_PARAM_HANDLER_ENTRIES
    _CUSTOM_PARAM_HANDLER_ENTRIES = {}
    _update_custom_param_handlers()
