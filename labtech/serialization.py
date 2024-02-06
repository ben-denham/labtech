"""Serialization/deserialization of tasks to/from JSON."""

from dataclasses import fields
from enum import Enum
from typing import cast, Dict, List, Optional, Type, Union

from frozendict import frozendict

from .types import Task, ResultMeta, is_task
from .exceptions import SerializationError
from .utils import ensure_dict_key_str

# Type to represent any value that can be handled by Python's default
# json encoder and decoder.
jsonable = Union[None, str, bool, float, int,
                 Dict[str, 'jsonable'], List['jsonable']]


class Serializer:

    def is_serialized_task(self, serialized: jsonable) -> bool:
        return isinstance(serialized, dict) and bool(serialized.get('_is_task', False))

    def serialize_task(self, task: Task) -> Dict[str, jsonable]:
        if not is_task(task):
            raise SerializationError(("serialize_task() must be called with a Task, "
                                      f"received: '{task}'"))

        serialized: Dict[str, jsonable] = {
            '_is_task': True,
            '__class__': self.serialize_class(task.__class__),
        }

        for field in fields(task):
            field_value = getattr(task, field.name)
            serialized_field = self.serialize_value(field_value)
            serialized[field.name] = serialized_field

        return serialized

    def deserialize_task(self, serialized: Dict[str, jsonable], *, result_meta: Optional[ResultMeta]) -> Task:
        if not self.is_serialized_task(serialized):
            raise SerializationError(("deserialize_task() must be called with a "
                                      f"serialized Task, received: '{serialized}'"))

        task_cls = self.deserialize_class(serialized['__class__'])
        cls_fields = {field.name: field for field in fields(task_cls)}

        params = {}
        for key, value in serialized.items():
            if key in {'_is_task', '__class__'}:
                continue

            if key not in cls_fields:
                cls_fullname = f'{task_cls.__module__}.{task_cls.__qualname__}'
                raise SerializationError((f"Serialized task contained field '{key}'"
                                          f"that is not present on Task class '{cls_fullname}'"))

            deserialized_value = self.deserialize_value(value)
            params[key] = deserialized_value

        task = task_cls(**params)
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
            return self.deserialize_task(cast(Dict[str, jsonable], value), result_meta=None)
        elif self.is_serialized_enum(value):
            return self.deserialize_enum(cast(Dict[str, jsonable], value))
        return value

    def is_serialized_enum(self, serialized: jsonable) -> bool:
        return isinstance(serialized, dict) and bool(serialized.get('_is_enum', False))

    def serialize_enum(self, value: Enum) -> jsonable:
        return {
            '_is_enum': True,
            '__class__': self.serialize_class(value.__class__),
            'name': value.name,
        }

    def deserialize_enum(self, serialized: Dict[str, jsonable]) -> Enum:
        enum_cls = self.deserialize_class(serialized['__class__'])
        return enum_cls[serialized['name']]

    def serialize_class(self, cls: Type) -> jsonable:
        return f'{cls.__module__}.{cls.__qualname__}'

    def deserialize_class(self, serialized_class: jsonable) -> Type:
        cls_module, cls_name = cast(str, serialized_class).rsplit('.', 1)
        module = __import__(cls_module, fromlist=[cls_name])
        return getattr(module, cls_name)
