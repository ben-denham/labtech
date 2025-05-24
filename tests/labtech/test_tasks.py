from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from enum import Enum
from typing import TYPE_CHECKING

import pytest
from frozendict import frozendict

import labtech
import labtech.tasks
from labtech.cache import BaseCache, NullCache, PickleCache
from labtech.exceptions import TaskError
from labtech.tasks import _RESERVED_ATTRS, find_tasks_in_param, immutable_param_value

if TYPE_CHECKING:
    from labtech.tasks import ParamScalar
    from labtech.types import ResultT, Storage, Task, TaskInfo


class _BadObject:
    """An object which is not supported in a task parameter."""

    pass


class _ExampleEnum(Enum):
    A = 1
    B = 2


@labtech.task
class ExampleTask:
    a: int

    def run(self) -> int:
        return self.a


@pytest.fixture(
    params=[
        None,
        True,
        False,
        'hello',
        'world',
        3.14,
        42,
        _ExampleEnum.A,
    ],
)
def scalar(request: pytest.FixtureRequest) -> ParamScalar:
    return request.param


class BadCache(BaseCache):
    """A pretend cache that returns fake values."""

    KEY_PREFIX = 'bad__'

    def save_result(self, storage: Storage, task: Task[ResultT], result: ResultT):
        raise NotImplementedError

    def load_result(self, storage: Storage, task: Task[ResultT]) -> ResultT:
        raise NotImplementedError


class TestTask:
    def test_defaults(self) -> None:
        @labtech.task
        class SimpleTask:
            def run(self) -> None:
                return None

        task = SimpleTask()
        task_info: TaskInfo = task._lt

        assert isinstance(task_info.cache, PickleCache)
        assert task_info.max_parallel is None
        assert task_info.mlflow_run is False
        assert task_info.orig_post_init is None

    def test_null_cache(self) -> None:
        @labtech.task(cache=None)
        class SimpleTask:
            def run(self) -> None:
                return None

        task = SimpleTask()
        task_info: TaskInfo = task._lt
        assert isinstance(task_info.cache, NullCache)

    def test_nondefault_cache(self) -> None:
        @labtech.task(cache=BadCache())
        class SimpleTask:
            def run(self) -> None:
                return None

        task = SimpleTask()
        task_info: TaskInfo = task._lt
        assert isinstance(task_info.cache, BadCache)

    @pytest.mark.parametrize('max_parallel', [None, 1, 2, 3])
    def test_max_parallel(self, max_parallel: int | None) -> None:
        @labtech.task(max_parallel=max_parallel)
        class SimpleTask:
            def run(self) -> None:
                pass

        task = SimpleTask()
        task_info: TaskInfo = task._lt
        assert task_info.max_parallel == max_parallel

    def test_mlflow_run(self) -> None:
        @labtech.task(mlflow_run=True)
        class SimpleTask:
            def run(self) -> None:
                pass

        task = SimpleTask()
        task_info: TaskInfo = task._lt
        assert task_info.mlflow_run is True

    def test_reserved_lt_attr(self) -> None:
        match = re.escape("Task type already defines reserved attribute '_lt'.")
        with pytest.raises(AttributeError, match=match):

            @labtech.task
            class SimpleTask:
                def run(self) -> None:
                    pass

                def _lt(self) -> None:
                    pass

                def _is_task(self) -> None:
                    pass

    @pytest.mark.parametrize('badattr', _RESERVED_ATTRS)
    def test_fail_reserved_attrs(self, badattr: str) -> None:
        class SimpleTaskBase:
            pass

        setattr(SimpleTaskBase, badattr, None)

        match = re.escape(f"Task type already defines reserved attribute '{badattr}'.")
        with pytest.raises(AttributeError, match=match):

            @labtech.task(mlflow_run=True)
            class SimpleTask(SimpleTaskBase):
                def run(self) -> None:
                    pass

    def test_stored_post_init(self) -> None:
        @labtech.task
        class SimpleTask:
            def post_init(self):
                return "It's me!"

            def run(self) -> None:
                pass

        task = SimpleTask()
        task_info: TaskInfo = task._lt
        assert task_info.orig_post_init is not None
        assert task_info.orig_post_init(task) == "It's me!"

    def test_filter_context(self) -> None:
        @labtech.task
        class SimpleTask:
            def filter_context(self, context):
                return {
                    'b': context['b'],
                }

            def run(self) -> None:
                pass

        task = SimpleTask()
        assert task.filter_context({'a': 1, 'b': 2}) == {'b': 2}

    def test_default_filter_context(self) -> None:
        @labtech.task
        class SimpleTask:
            def run(self) -> None:
                pass

        task = SimpleTask()
        assert task.filter_context({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    def test_frozen(self) -> None:
        @labtech.task
        class SimpleTask:
            a: int

            def run(self) -> None:
                return None

        task = SimpleTask(a=1)

        # Check the dataclass is now frozen
        with pytest.raises(FrozenInstanceError):
            task.a = 2

    def test_order(self) -> None:
        @labtech.task
        class SimpleTask:
            a: int
            b: str

            def run(self) -> None:
                return None

        task1 = SimpleTask(a=1, b='hello')
        task2 = SimpleTask(b='hello', a=2)
        task3 = SimpleTask(a=1, b='zzz')

        assert task1 < task2
        assert task2 > task3
        assert task1 <= task3
        assert task1 != task2
        assert task1 == task1

    def test_fail_no_run(self) -> None:
        match = re.escape("Task type 'SimpleTask' must define a 'run' method")
        with pytest.raises(NotImplementedError, match=match):

            @labtech.task
            class SimpleTask:
                pass

    def test_fail_noncallable_run(self) -> None:
        match = re.escape("Task type 'SimpleTask' must define a 'run' method")
        with pytest.raises(NotImplementedError, match=match):

            @labtech.task
            class SimpleTask:
                run: int

    def test_post_init_missing_dunder(self) -> None:
        match = re.escape(
            "Task type already defines reserved attribute '__post_init__'."
        )
        with pytest.raises(AttributeError, match=match):

            @labtech.task
            class SimpleTask:
                def __post_init__(self):
                    pass

                def run(self) -> None:
                    pass

    def test_inheritance(self) -> None:
        @labtech.task
        class SimpleTask:
            def run(self) -> None:
                pass

        @labtech.task
        class SubTask(SimpleTask):
            def run(self) -> None:
                pass

        # Check we don't get an error trying to do this.
        SubTask()


class TestImmutableParamValue:
    def test_empty_list(self) -> None:
        assert immutable_param_value('hello', []) == ()

    def test_empty_dict(self) -> None:
        assert immutable_param_value('hello', {}) == frozendict()

    def test_list(self) -> None:
        assert immutable_param_value('hello', [1, 2, 3]) == (1, 2, 3)

    def test_dict(self) -> None:
        assert immutable_param_value('hello', {'a': 1, 'b': 2}) == frozendict(
            {'a': 1, 'b': 2}
        )

    def test_frozendict(self) -> None:
        assert immutable_param_value(
            'hello', frozendict({'a': 1, 'b': 2})
        ) == frozendict({'a': 1, 'b': 2})

    def test_tuple(self) -> None:
        assert immutable_param_value('hello', (1, 2, 3)) == (1, 2, 3)

    def test_nested_list(self) -> None:
        assert immutable_param_value('hello', [1, [2, 3], 4]) == (1, (2, 3), 4)

    def test_nested_dict(self) -> None:
        assert immutable_param_value('hello', {'a': 1, 'b': {'c': 2}}) == frozendict(
            {'a': 1, 'b': frozendict({'c': 2})}
        )

    def test_multiple_nesting(self) -> None:
        assert immutable_param_value('hello', {'a': [1, {'b': 2}]}) == frozendict(
            {'a': (1, frozendict({'b': 2}))}
        )

    def test_nested_list_dict(self) -> None:
        assert immutable_param_value('hello', [1, {'a': 2}]) == (
            1,
            frozendict({'a': 2}),
        )

    def test_scalar(self, scalar: ParamScalar) -> None:
        assert immutable_param_value('hello', scalar) is scalar

    def test_unhandled(self) -> None:
        with pytest.raises(
            TaskError, match="Unsupported type '_BadObject' in parameter value 'hello'."
        ):
            immutable_param_value('hello', _BadObject())

    def test_multiple_nested_error(self) -> None:
        match = re.escape(
            """Unsupported type '_BadObject' in parameter value 'hello["b"][2]["c"]'."""
        )
        with pytest.raises(TaskError, match=match):
            immutable_param_value('hello', {'a': 1, 'b': (1, 2, {'c': _BadObject()})})


class TestFindTasksInParam:
    def test_scalar(self, scalar: ParamScalar) -> None:
        assert find_tasks_in_param(scalar) == []

    def test_task(self) -> None:
        task = ExampleTask(1)
        assert find_tasks_in_param(task) == [task]

    def test_list(self) -> None:
        task = ExampleTask(1)
        assert find_tasks_in_param([task]) == [task]
        assert find_tasks_in_param([]) == []

    def test_dict(self) -> None:
        task = ExampleTask(1)
        assert find_tasks_in_param({'a': task}) == [task]
        assert find_tasks_in_param({}) == []

    def test_nested_list(self) -> None:
        task1 = ExampleTask(1)
        task2 = ExampleTask(2)
        assert find_tasks_in_param([task1, [task2]]) == [task1, task2]

    def test_nested_dict(self) -> None:
        task1 = ExampleTask(1)
        task2 = ExampleTask(2)
        assert find_tasks_in_param({'a': task1, 'b': {'c': task2}}) == [task1, task2]

    def test_tuple(self) -> None:
        task = ExampleTask(1)
        assert find_tasks_in_param((task,)) == [task]

    def test_frozen_dict(self) -> None:
        task = ExampleTask(1)
        assert find_tasks_in_param(frozendict({'a': task})) == [task]

    def test_searched_coll_ids(self) -> None:
        task1 = ExampleTask(1)
        task2 = ExampleTask(2)
        assert find_tasks_in_param([task1, task2], searched_coll_ids={id(task1)}) == [
            task2
        ]

    def test_unhandled(self) -> None:
        match = re.escape(
            'Unexpected type _BadObject encountered in task parameter value.'
        )
        with pytest.raises(TaskError, match=match):
            find_tasks_in_param(_BadObject())
