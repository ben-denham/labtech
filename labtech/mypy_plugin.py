"""Mypy plugin for classes decorated with `labtech.task`."""

from typing import Type
import mypy.plugins.dataclasses
from mypy.plugin import Plugin, ClassDefContext
from mypy.plugins.common import add_attribute_to_class
from mypy.types import Instance, LiteralType, AnyType, TypeOfAny, CallableType, NoneType, UnionType
from mypy.nodes import ArgKind

task_makers = {'labtech.tasks.task'}
"""Set of decorator functions that return "task" classes."""


def task_tag_callback(ctx: ClassDefContext):
    """Add attributes to a "task" class."""

    results_map_type = Instance(
        typ=ctx.api.named_type('builtins.dict').type,
        args=[
            ctx.api.named_type('labtech.types.Task'),
            AnyType(TypeOfAny.explicit),
        ],
    )

    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='_lt',
        typ=ctx.api.named_type('labtech.types.TaskInfo'),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='_is_task',
        typ=LiteralType(value=True, fallback=ctx.api.named_type('builtins.bool')),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='_set_results_map',
        typ=CallableType(
            arg_types=[results_map_type],
            arg_kinds=[ArgKind.ARG_POS],
            arg_names=['result_map'],
            ret_type=NoneType(),
            fallback=ctx.api.named_type('builtins.function'),
        ),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='_results_map',
        typ=UnionType([NoneType(), results_map_type]),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='cache_key',
        typ=ctx.api.named_type('builtins.str'),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='result',
        typ=AnyType(TypeOfAny.explicit),
    )


class LabtechPlugin(Plugin):
    """Add attributes to "task" classes."""

    def get_class_decorator_hook(self, fullname: str):
        if fullname in task_makers:
            return task_tag_callback


def plugin(version: str) -> Type[LabtechPlugin]:
    # Task types should be handled like other dataclasses
    mypy.plugins.dataclasses.dataclass_makers.update(task_makers)
    return LabtechPlugin
