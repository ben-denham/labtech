from pprint import pprint
from IPython.display import Markdown

from collections import defaultdict
from dataclasses import dataclass, Field, fields
from textwrap import indent
from typing import Dict, Sequence, Type, GenericAlias, get_origin, get_args

from frozendict import frozendict

from labtech.types import Task
from labtech.tasks import find_tasks_in_param

@dataclass(frozen=True)
class TaskRelationshipKey:
    from_param_name: str
    to_task_type: Type[Task]

@dataclass(frozen=True)
class TaskRelationshipInfo:
    multi_cardinality: bool

class TaskStructure:

    def __init__(self):
        self.task_type_to_relationships: Dict[Type[Task], Dict[TaskRelationshipKey, TaskRelationshipInfo]] = {}

    def add_task_type(self, task_type: Type[Task]):
        if task_type not in self.task_type_to_relationships:
            self.task_type_to_relationships[task_type] = {}

    def add_relationship(self, *, from_task_type: Type[Task], from_param_field: Field, to_task_type: Type[Task]):
        key = TaskRelationshipKey(
            from_param_name=from_param_field.name,
            to_task_type=to_task_type,
        )
        info = TaskRelationshipInfo(
            # TODO: Explain
            multi_cardinality=(
                from_param_field.type in {
                    list,
                    tuple,
                    dict,
                    frozendict
                }
                or
                get_origin(from_param_field.type) in {
                    list,
                    tuple,
                    dict,
                    frozendict
                }
            )
        )
        rels = self.task_type_to_relationships[from_task_type]
        if key not in rels:
            rels[key] = info
        else:
            old_info = rels[key]
            rels[key] = TaskRelationshipInfo(
                multi_cardinality=(old_info.multi_cardinality or info.multi_cardinality),
            )

    @classmethod
    def build(cls, tasks: Sequence[Task]):
        task_structure = cls()

        found_tasks = list(tasks)
        while True:
            try:
                task = found_tasks.pop(0)
            except IndexError:
                # No more found_tasks, exit the loop
                return task_structure

            task_structure.add_task_type(type(task))

            for field in fields(task):
                param_value = getattr(task, field.name)
                sub_tasks = find_tasks_in_param(param_value)

                found_tasks += sub_tasks
                for sub_task in sub_tasks:
                    task_structure.add_relationship(
                        from_task_type=type(task),
                        from_param_field=field,
                        to_task_type=type(sub_task),
                    )


task_structure = TaskStructure.build([evaluation_task])
pprint(task_structure.task_type_to_relationships)


def format_type(t):
    origin = get_origin(t)
    if origin is None:
        return t.__name__
    else:
        args_str = ', '.join([format_type(arg_t) for arg_t in get_args(t)])
        return f'{origin.__name__}[{args_str}]'

def diagram_task_structure(task_structure, *, direction='BT'):
    classes = '\n\n'.join([
        (
            '\n'.join([
                f'class {task_type.__name__}',
                *[
                    f'{task_type.__name__} : {format_type(field.type)} {field.name}'
                    for field in fields(task_type)
                    if field
                ]
            ])
        )
        for task_type in task_structure.task_type_to_relationships.keys()
    ])
    relationships = '\n\n'.join([
        '\n'.join([
            (f'{from_task_type.__name__} --o "{"many" if rel_info.multi_cardinality else ""}" '
             f'{rel_key.to_task_type.__name__}: {rel_key.from_param_name}')
            for rel_key, rel_info in rels.items()
        ])
        for from_task_type, rels in task_structure.task_type_to_relationships.items()
        if rels
    ])
    return f'''```mermaid
classDiagram
    direction {direction}

{indent(classes, ' ' * 4)}


{indent(relationships, ' ' * 4)}
    ```'''

print(diagram_task_structure(task_structure))
display(Markdown(diagram_task_structure(task_structure, direction='BT')))
