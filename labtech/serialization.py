"""Serialization/deserialization of tasks to/from JSON."""
from __future__ import annotations

from dataclasses import fields
from enum import Enum
from typing import TYPE_CHECKING, cast

from frozendict import frozendict

from .exceptions import SerializationError
from .types import is_task
from .utils import ensure_dict_key_str

if TYPE_CHECKING:
    from .types import ResultMeta, Task

# Type to represent any value that can be handled by Python's default
# json encoder and decoder.
jsonable = None | str | bool | float | int | dict[str, 'jsonable'] | list['jsonable']

class Serializer:

    def is_serialized_task(self, serialized: jsonable) -> bool:
        return isinstance(serialized, dict) and bool(serialized.get('_is_task', False))

    def serialize_task(self, task: Task) -> dict[str, jsonable]:
        if not is_task(task):
            raise SerializationError(("serialize_task() must be called with a Task, "
                                      f"received: '{task}'"))

        serialized: dict[str, jsonable] = {
            '_is_task': True,
            '__class__': self.serialize_class(task.__class__),
        }

        # Include the code version in the serialized task, which is
        # used to generate the hash that determines task equality.
        if task.code_version is not None:
            serialized['_code_version'] = task.code_version

        for field in fields(task):
            field_value = getattr(task, field.name)
            serialized_field = self.serialize_value(field_value)
            serialized[field.name] = serialized_field

        return serialized

    def deserialize_task(self, serialized: dict[str, jsonable], *, result_meta: ResultMeta | None) -> Task:
        if not self.is_serialized_task(serialized):
            raise SerializationError(("deserialize_task() must be called with a "
                                      f"serialized Task, received: '{serialized}'"))

        task_cls = self.deserialize_class(serialized['__class__'])
        cls_fields = {field.name: field for field in fields(task_cls)}

        params = {}
        for key, value in serialized.items():
            if key in {'_is_task', '_code_version', '__class__'}:
                continue

            if key not in cls_fields:
                cls_fullname = f'{task_cls.__module__}.{task_cls.__qualname__}'
                raise SerializationError((f"Serialized task contained field '{key}' "
                                          f"that is not present on Task class '{cls_fullname}'"))

            deserialized_value = self.deserialize_value(value)
            params[key] = deserialized_value

        task = task_cls(**params)
        task._set_code_version(serialized.get('_code_version', None))
        task._set_result_meta(result_meta)
        return task

    def serialize_value(self, value) -> jsonable:
        if is_task(value):
            return self.serialize_task(value)
        elif isinstance(value, tuple):
            return [self.serialize_value(item) for item in value]
        elif isinstance(value, frozendict):
            return {
                ensure_dict_key_str(key, exception_type=SerializationError): self.serialize_value(value)
                for key, value in value.items()
            }
        elif isinstance(value, Enum):
            return self.serialize_enum(value)
        elif ((value is None)
              or isinstance(value, str)
              or isinstance(value, bool)
              or isinstance(value, float)
              or isinstance(value, int)):
            return value
        raise SerializationError((f"Cannot serialize value: '{value}'. Please check "
                                  "that your task's parameters only use supported types."))

    def deserialize_value(self, value: jsonable):
        if self.is_serialized_task(value):
            return self.deserialize_task(cast('dict[str, jsonable]', value), result_meta=None)
        elif isinstance(value, list):
            return tuple([self.deserialize_value(item) for item in value])
        elif isinstance(value, dict):
            return frozendict({
              ensure_dict_key_str(k, exception_type=SerializationError): self.deserialize_value(v)
              for k, v in value.items()
            })
        elif self.is_serialized_enum(value):
            return self.deserialize_enum(cast('dict[str, jsonable]', value))
        return value

    def is_serialized_enum(self, serialized: jsonable) -> bool:
        return isinstance(serialized, dict) and bool(serialized.get('_is_enum', False))

    def serialize_enum(self, value: Enum) -> jsonable:
        return {
            '_is_enum': True,
            '__class__': self.serialize_class(value.__class__),
            'name': value.name,
        }

    def deserialize_enum(self, serialized: dict[str, jsonable]) -> Enum:
        enum_cls = cast('type[Enum]', self.deserialize_class(serialized['__class__']))
        name = cast('str', serialized['name'])
        return enum_cls[name]

    def serialize_class(self, cls: type) -> jsonable:
        if '<locals>' in cls.__qualname__:
            raise SerializationError(
                (f'Unable to serialize class "{cls.__qualname__}" because it was defined in a function. '
                 'Please ensure all task and parameter classes are defined at the top-level of a module.')
            )

        if '>' in cls.__qualname__:
            # This should never happen, but is here for safety.
            raise SerializationError('Unexpected ">" in class __qualname__')

        # Handle nested classes by splitting class nesting path by ">".
        return f'{cls.__module__}.{cls.__qualname__.replace(".", ">")}'

    def deserialize_class(self, serialized_class: jsonable) -> type:
        cls_module, cls_qualname = cast('str', serialized_class).rsplit('.', 1)
        cls_name_parts = cls_qualname.split('>')
        module = __import__(cls_module, fromlist=[cls_name_parts[0]])

        cls = getattr(module, cls_name_parts[0])
        for part in cls_name_parts[1:]:
            # Navigate to nested class.
            cls = getattr(cls, part)

        return cls
