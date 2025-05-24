"""Mypy plugin for classes decorated with `labtech.task`."""
from __future__ import annotations

from typing import TYPE_CHECKING

import mypy.plugins.dataclasses
from mypy.nodes import COVARIANT, ArgKind
from mypy.plugin import Plugin
from mypy.plugins.common import add_attribute_to_class
from mypy.types import AnyType, CallableType, Instance, LiteralType, NoneType, TypeOfAny, TypeVarId, TypeVarType, UnionType

if TYPE_CHECKING:
    from mypy.plugin import ClassDefContext

task_makers = {'labtech.tasks.task'}
"""Set of decorator functions that return "task" classes."""


def task_tag_callback(ctx: ClassDefContext):
    """Add attributes to a "task" class."""
    covariant_result_t_type = TypeVarType(
        name='CovariantResultT',
        fullname=f'{ctx.cls.info.fullname}.CovariantResultT',
        id=TypeVarId(-1, namespace=f"{ctx.cls.info.fullname}"),
        values=[],
        upper_bound=ctx.api.named_type('builtins.object'),
        variance=COVARIANT,
        default=AnyType(TypeOfAny.from_omitted_generics),
    )
    results_map_type = Instance(
        typ=ctx.api.named_type('builtins.dict').type,
        args=[
            ctx.api.named_type('labtech.types.Task', [
                covariant_result_t_type
            ]),
            covariant_result_t_type,
        ],
    )
    context_type = Instance(
        typ=ctx.api.named_type('builtins.dict').type,
        args=[
            ctx.api.named_type('builtins.str'),
            AnyType(TypeOfAny.explicit),
        ],
    )

    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='current_code_version',
        typ=UnionType([NoneType(), ctx.api.named_type('builtins.str')]),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='code_version',
        typ=UnionType([NoneType(), ctx.api.named_type('builtins.str')]),
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
        name='_cache_key',
        typ=UnionType([NoneType(), ctx.api.named_type('builtins.str')]),
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
        typ=covariant_result_t_type,
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='set_context',
        typ=CallableType(
            arg_types=[context_type],
            arg_kinds=[ArgKind.ARG_POS],
            arg_names=['context'],
            ret_type=NoneType(),
            fallback=ctx.api.named_type('builtins.function'),
        ),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='filter_context',
        typ=CallableType(
            arg_types=[context_type],
            arg_kinds=[ArgKind.ARG_POS],
            arg_names=['context'],
            ret_type=context_type,
            fallback=ctx.api.named_type('builtins.function'),
        ),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='context',
        typ=UnionType([NoneType(), context_type])
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='_set_result_meta',
        typ=CallableType(
            arg_types=[ctx.api.named_type('labtech.types.ResultMeta')],
            arg_kinds=[ArgKind.ARG_POS],
            arg_names=['result_meta'],
            ret_type=NoneType(),
            fallback=ctx.api.named_type('builtins.function'),
        ),
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='result_meta',
        typ=UnionType([NoneType(), ctx.api.named_type('labtech.types.ResultMeta')])
    )
    add_attribute_to_class(
        api=ctx.api,
        cls=ctx.cls,
        name='runner_options',
        typ=CallableType(
            arg_types=[],
            arg_kinds=[],
            arg_names=[],
            ret_type=Instance(
                typ=ctx.api.named_type('builtins.dict').type,
                args=[
                    ctx.api.named_type('builtins.str'),
                    AnyType(TypeOfAny.explicit),
                ],
            ),
            fallback=ctx.api.named_type('builtins.function'),
        ),
    )


class LabtechPlugin(Plugin):
    """Add attributes to "task" classes."""

    def get_class_decorator_hook(self, fullname: str):
        if fullname in task_makers:
            return task_tag_callback


def plugin(version: str) -> type[LabtechPlugin]:
    # Task types should be handled like other dataclasses
    mypy.plugins.dataclasses.dataclass_makers.update(task_makers)
    return LabtechPlugin
