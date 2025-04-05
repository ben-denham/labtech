"""Test a set of tasks packed with usage of features end-to-end.
Loosely based on tasks from the tutorial."""

from tempfile import TemporaryDirectory
from typing import Any, Protocol

import pytest

import labtech
from labtech.types import Task


@labtech.task(cache=None)
class ClassifierTask:
    n_estimators: int

    def run(self) -> dict:
        return {'n_estimators': self.n_estimators}


class ExperimentTask(Protocol):
    dataset_key: str

    def run(self) -> dict:
        pass


@labtech.task
class ClassifierExperiment(ExperimentTask):
    classifier_task: ClassifierTask
    dataset_key: str

    def filter_context(self, context: dict[str, Any]) -> dict[str, Any]:
        # Filter context to a single dataset.
        return {
            'DATASETS': {
                self.dataset_key: context['DATASETS'][self.dataset_key],
            },
        }

    def run(self) -> dict:
        # Check filtering was successful:
        assert len(self.context['DATASETS']) == 1

        return {
            'dataset': self.context['DATASETS'][self.dataset_key],
            'classifier': self.classifier_task.result,
        }


@labtech.task
class WrappingExperiment(ExperimentTask):
    experiment: ExperimentTask

    @property
    def dataset_key(self):
        return self.experiment.dataset_key

    def run(self) -> dict:
        return {
            'inner_experiment': self.experiment.result
        }


@labtech.task
class ExperimentEvaluationTask:
    experiments: list[ExperimentTask]

    def run(self) -> dict:
        return {
            str(i): experiment.result
            for i, experiment in enumerate(self.experiments, start=1)
        }


@pytest.fixture
def context() -> dict[str, Any]:
    return {
        'DATASETS': {
            'a': 'aaa',
            'b': 'bbb',
        },
    }


@pytest.fixture
def evaluation_task(context: dict[str, Any]):
    classifier_tasks = [
        ClassifierTask(
            n_estimators=n_estimators,
        )
        for n_estimators in range(1, 3)
    ]
    classifier_experiments = [
        ClassifierExperiment(
            classifier_task=classifier_task,
            dataset_key=dataset_key,
        )
        for dataset_key in context['DATASETS'].keys()
        for classifier_task in classifier_tasks
    ]
    wrapping_experiments = [
        WrappingExperiment(
            experiment=classifier_experiment,
        )
        for classifier_experiment in classifier_experiments
    ]
    evaluation_task = ExperimentEvaluationTask(
        experiments=[
            *classifier_experiments,
            *wrapping_experiments,
        ]
    )
    return evaluation_task



class TestE2E:

    @pytest.mark.parametrize("max_workers", [1, 4, None])
    @pytest.mark.parametrize("runner_backend", ['serial', 'fork', 'spawn'])
    def test_e2e(self, max_workers: int, runner_backend: str, context: dict[str, Any], evaluation_task: Task) -> None:
        with TemporaryDirectory() as storage_dir:
            lab = labtech.Lab(
                storage=storage_dir,
                context=context,
                max_workers=max_workers,
                runner_backend=runner_backend,
            )
            evaluation_result = lab.run_task(evaluation_task)
        assert evaluation_result == {
            '1': {'dataset': 'aaa', 'classifier': {'n_estimators': 1}},
            '2': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}},
            '3': {'dataset': 'bbb', 'classifier': {'n_estimators': 1}},
            '4': {'dataset': 'bbb', 'classifier': {'n_estimators': 2}},
            '5': {'inner_experiment': {'dataset': 'aaa', 'classifier': {'n_estimators': 1}}},
            '6': {'inner_experiment': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}}},
            '7': {'inner_experiment': {'dataset': 'bbb', 'classifier': {'n_estimators': 1}}},
            '8': {'inner_experiment': {'dataset': 'bbb', 'classifier': {'n_estimators': 2}}},
        }
