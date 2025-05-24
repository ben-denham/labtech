from __future__ import annotations

from dataclasses import dataclass, fields
from textwrap import indent
from typing import TYPE_CHECKING, get_args, get_origin, get_type_hints

from .tasks import find_tasks_in_param
from .types import is_task
from .utils import is_ipython

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from .types import Task


@dataclass(frozen=True)
class TaskRelKey:
    from_param_name: str
    to_task_type: type[Task]


@dataclass(frozen=True)
class TaskRelInfo:
    multi_cardinality: bool


class TaskStructure:
    """Records the structure of task types and dependencies/relationships
    between task types."""

    def __init__(self) -> None:
        self.task_type_to_rels: dict[type[Task], dict[TaskRelKey, TaskRelInfo]] = {}

    def add_task_type(self, task_type: type[Task]):
        self.task_type_to_rels.setdefault(task_type, {})

    def add_relationship(self, *, from_task_type: type[Task], from_param_name: str,
                         to_task_type: type[Task], multi_cardinality: bool):
        key = TaskRelKey(
            from_param_name=from_param_name,
            to_task_type=to_task_type,
        )
        info = TaskRelInfo(
            multi_cardinality=multi_cardinality,
        )
        rels = self.task_type_to_rels[from_task_type]
        if key not in rels:
            rels[key] = info
        else:
            old_info = rels[key]
            rels[key] = TaskRelInfo(
                # If a task is sometimes observed with multi_cardinality
                # and sometimes not, then mark it as multi_cardinality.
                multi_cardinality=(old_info.multi_cardinality or info.multi_cardinality),
            )

    @classmethod
    def build(cls, tasks: Sequence[Task]) -> TaskStructure:
        task_structure = cls()

        # Start by working through the provided tasks
        found_tasks = list(tasks)
        while True:
            try:
                task = found_tasks.pop(0)
            except IndexError:
                # No more found_tasks, exit the loop
                return task_structure

            # Ensure we have recorded this type of task
            task_structure.add_task_type(type(task))

            # Search for sub_tasks in each param/field of the task
            for field in fields(task):
                param_value = getattr(task, field.name)
                sub_tasks = find_tasks_in_param(param_value)

                # Add a relationship for each found sub_task
                for sub_task in sub_tasks:
                    task_structure.add_relationship(
                        from_task_type=type(task),
                        from_param_name=field.name,
                        to_task_type=type(sub_task),
                        multi_cardinality=(not is_task(param_value)),
                    )
                # Add the tasks to the list of tasks to work through
                found_tasks += sub_tasks


def format_type(t: type | str | Any) -> str:
    """Format the given type as a string without name qualification.
    For type hints that aren't actually types, return the string
    representation."""
    if not isinstance(t, type):
        return str(t)

    origin = get_origin(t)
    if origin is None:
        # For non-generic types, just return the type's name
        return t.__name__
    else:
        # For generic types, format each type argument
        args_str = ', '.join([format_type(arg_t) for arg_t in get_args(t)])
        return f'{origin.__name__}[{args_str}]'


def diagram_task_type(task_type: type[Task]) -> str:
    run_return_type = get_type_hints(task_type.run).get('return')
    run_return = (
        '' if run_return_type is None else f' {format_type(run_return_type)}'
    )
    return '\n'.join([
        f'class {format_type(task_type)}',
        *[
            f'{format_type(task_type)} : {format_type(field.type)} {field.name}'
            for field in fields(task_type)
            if field
        ],
        f'{format_type(task_type)} : run(){run_return}'
    ])


def diagram_task_relationship(from_task_type: type[Task], relationships: dict[TaskRelKey, TaskRelInfo]) -> str:

    def format_many(multi_cardinality: bool) -> str:
        if multi_cardinality:
            return '"many" '
        return ''

    return '\n'.join([
        (f'{format_type(from_task_type)} <-- {format_many(rel_info.multi_cardinality)}'
         f'{format_type(rel_key.to_task_type)}: {rel_key.from_param_name}')
        for rel_key, rel_info in relationships.items()
    ])


def diagram_task_structure(task_structure: TaskStructure, *, direction: str) -> str:
    indentation = ' ' * 4
    classes = '\n\n'.join([
        diagram_task_type(task_type)
        for task_type in task_structure.task_type_to_rels.keys()
    ])
    relationships = '\n\n'.join([
        diagram_task_relationship(from_task_type, relationships)
        for from_task_type, relationships in task_structure.task_type_to_rels.items()
        if relationships
    ])
    return (
        'classDiagram\n'
        + indent(f'direction {direction}', indentation) + '\n\n'
        + indent(classes, indentation) + '\n\n\n'
        + indent(relationships, indentation)
    )


def build_task_diagram(tasks: Sequence[Task], *, direction: str = 'BT') -> str:
    """Returns a [Mermaid diagram](https://mermaid.js.org/syntax/classDiagram.html)
    representing the task types and dependencies of the given tasks.

    Each task type lists its parameters (with their return types) and
    its run method (with its return type).

    Arrows between task types point from a dependency task type to the
    task type that depends on it, and are labelled with the dependent
    task's parameter that references the dependency task type.

    Args:
        tasks: A collection of tasks to diagram.
        direction: Direction that task types should be laid out, from dependent
            tasks to their dependencies. One of:

            * `'BT'` (bottom-to-top)
            * `'TB'` (top-to-bottom)
            * `'RL'` (right-to-left)
            * `'LR'` (left-to-right)

    """
    return diagram_task_structure(
        TaskStructure.build(tasks),
        direction=direction,
    )


def display_task_diagram(tasks: Sequence[Task], **kwargs) -> None:
    """Displays a [Mermaid diagram](https://mermaid.js.org/syntax/classDiagram.html)
    representing the task types and dependencies of the given tasks.

    If IPython is available (e.g. the code is being run from a Jupyter
    notebook), the diagram will be displayed as a Markdown `mermaid`
    code block, which will be rendered as a Mermaid diagram from
    [JupyterLab 4.1 and Notebook 7.1](https://blog.jupyter.org/jupyterlab-4-1-and-notebook-7-1-are-here-20bfc3c10217).

    Because Markdown may render arbitrary HTML, you should only
    diagram tasks that you trust.

    Accepts the same arguments as
    [build_task_diagram][labtech.diagram.build_task_diagram].

    """
    diagram = build_task_diagram(tasks, **kwargs)
    if is_ipython():
        from IPython.display import Markdown, display
        display(Markdown(f'```mermaid\n{diagram}\n```'))
    else:
        print(diagram)
