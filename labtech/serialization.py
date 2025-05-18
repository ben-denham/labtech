"""Serialization/deserialization of tasks to/from JSON."""

from abc import ABC, abstractmethod
from dataclasses import fields
from enum import Enum
from typing import Any, Optional, Sequence, Type, Union, cast

from frozendict import frozendict

from .exceptions import SerializationError
from .types import ResultMeta, Task, is_task
from .utils import ensure_dict_key_str

# Type to represent any value that can be handled by Python's default
# json encoder and decoder.
jsonable = Union[None, str, bool, float, int,
                 dict[str, 'jsonable'], list['jsonable']]


class CustomSerializer(ABC):
    """Base class for custom serializers that can convert complex
    objects into JSON-compatible representations."""

    @abstractmethod
    def handles(self, value: Any) -> bool:
        """Returns True if value should be serialized by this
        serializer."""

    @abstractmethod
    def serialize(self, serializer: 'Serializer', value: Any) -> jsonable:
        """Convert value into a JSON-compatible representation
        composed only of dictionaries, lists, strings, numbers and
        `None`.

        Also receives the full Serializer, which can be used to call
        `serializer.serialize_value()` to serialize nested elements
        within value."""

    @abstractmethod
    def deserialize(self, serializer: 'Serializer', serialized: jsonable) -> Any:
        """Convert the serialized representation returned by
        serialize() back into the original value.

        Also receives the full Serializer, which can be used to call
        `serializer.deserialize_value()` to deserialize nested elements
        within serialized."""


class Serializer:
    """Serializer for producing serialized JSON representations of
    Task objects, and deserializing JSON back into Task objects."""

    def __init__(self, custom_serializer_classes: Optional[Sequence[Type[CustomSerializer]]] = None):
        """
        Args:
            custom_serializer_classes: A list of classes that inherit from
                [CustomSerializer][labtech.serialization.CustomSerializer] that
                extend the types of task parameters that can be serialized. When
                a value is serialized, the `handle()` method of an instance of
                each custom_serializer_class is called in the provided order to
                determine whether it should be used to serialize that value.
                Custom serializers are applied before default serialization is
                applied.

        """
        self.custom_serializers = [
            custom_serializer_class() for custom_serializer_class
            in ([] if custom_serializer_classes is None else custom_serializer_classes)
        ]

    def _is_serialized_custom(self, serialized: jsonable) -> bool:
        return isinstance(serialized, dict) and bool(serialized.get('_is_custom', False))

    def _serialize_custom(self, custom_serializer: CustomSerializer, value: Any) -> dict[str, jsonable]:
        return {
            '_is_custom': True,
            '__class__': self.serialize_class(custom_serializer.__class__),
            'value': custom_serializer.serialize(
                serializer=self,
                value=value,
            ),
        }

    def _deserialize_custom(self, serialized: dict[str, jsonable]) -> Any:
        if not self._is_serialized_custom(serialized):
            raise SerializationError(("deserialize_custom() must be called with a "
                                      f"serialized custom value, received: '{serialized}'"))

        custom_serializer = self.deserialize_class(serialized['__class__'])()
        return custom_serializer.deserialize_value(
            serializer=self,
            serialized=serialized['value'],
        )

    def _is_serialized_enum(self, serialized: jsonable) -> bool:
        return isinstance(serialized, dict) and bool(serialized.get('_is_enum', False))

    def _serialize_enum(self, value: Enum) -> jsonable:
        return {
            '_is_enum': True,
            '__class__': self.serialize_class(value.__class__),
            'name': value.name,
        }

    def _deserialize_enum(self, serialized: dict[str, jsonable]) -> Enum:
        enum_cls = self.deserialize_class(serialized['__class__'])
        return enum_cls[serialized['name']]


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

    def deserialize_task(self, serialized: dict[str, jsonable], *, result_meta: Optional[ResultMeta]) -> Task:
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

    def serialize_value(self, value: Any) -> jsonable:
        for custom_serializer in self.custom_serializers:
            if custom_serializer.handles(value):
                return self._serialize_custom(custom_serializer, value)

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
            return self._serialize_enum(value)
        elif ((value is None)
              or isinstance(value, str)
              or isinstance(value, bool)
              or isinstance(value, float)
              or isinstance(value, int)):
            return value
        raise SerializationError((f"Cannot serialize value: '{value}'. Please check "
                                  "that your task's parameters only use supported types."))

    def deserialize_value(self, value: jsonable):
        if self._is_serialized_custom(value):
            return self._deserialize_custom(cast(dict[str, jsonable], value))
        elif self.is_serialized_task(value):
            return self.deserialize_task(cast(dict[str, jsonable], value), result_meta=None)
        elif isinstance(value, list):
            return tuple([self.deserialize_value(item) for item in value])
        elif isinstance(value, dict):
            return frozendict({
              ensure_dict_key_str(k, exception_type=SerializationError): self.deserialize_value(v)
              for k, v in value.items()
            })
        elif self._is_serialized_enum(value):
            return self._deserialize_enum(cast(dict[str, jsonable], value))
        return value

    def serialize_class(self, cls: Type) -> jsonable:
        return f'{cls.__module__}.{cls.__qualname__}'

    def deserialize_class(self, serialized_class: jsonable) -> Type:
        cls_module, cls_name = cast(str, serialized_class).rsplit('.', 1)
        module = __import__(cls_module, fromlist=[cls_name])
        return getattr(module, cls_name)
