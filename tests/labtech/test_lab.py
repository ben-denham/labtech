from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

import labtech
from labtech.lab import Lab

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def lab(tmp_path: Path) -> Lab:
    return Lab(storage=tmp_path, max_workers=1)


class TestLab:
    class TestRunTasks:
        def test_no_tasks(self, lab: Lab) -> None:
            results = lab.run_tasks(tasks=[])
            assert results == {}

        def test_simple_task(self, lab: Lab) -> None:
            results = lab.run_tasks(tasks=[_SimpleTask(a=1)])
            assert results == {_SimpleTask(a=1): 1}

        def test_noparam_task(self, lab: Lab) -> None:
            results = lab.run_tasks(tasks=[_NoParamTask()])
            assert results == {_NoParamTask(): 1}

        def test_duplicated_task(self, lab: Lab) -> None:
            """Tests both duplicate parent tasks and duplicated child tasks."""
            tasks = [
                _ParentTask(child=_NoParamTask()),
                _ParentTask(child=_NoParamTask()),
            ]
            results = lab.run_tasks(tasks)

            assert results == {
                _ParentTask(child=_NoParamTask()): 1,
            }

            for task in tasks:
                assert task.result_meta is not None
                assert task.child.result_meta is not None


@labtech.task(cache=None)
class _SimpleTask:
    a: int

    def run(self) -> int:
        return self.a


@labtech.task(cache=None)
class _NoParamTask:
    def run(self) -> Literal[1]:
        return 1


@labtech.task(cache=None)
class _ParentTask:
    child: _NoParamTask

    def run(self) -> int:
        return self.child.result
