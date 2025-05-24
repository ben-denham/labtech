"""Test a set of tasks packed with usage of features end-to-end.
Loosely based on tasks from the tutorial."""

import platform
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import pytest
import ray

import labtech
from labtech.exceptions import RunnerError
from labtech.runners.ray import RayRunnerBackend

if TYPE_CHECKING:
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


class Evaluation(TypedDict):
    task: 'Task'
    expected_result: Any


def basic_evaluation(context: dict[str, Any]) -> Evaluation:
    """Evaluation of a standard setup of multiple levels of dependency."""
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
    return Evaluation(
        task=evaluation_task,
        expected_result={
            '1': {'dataset': 'aaa', 'classifier': {'n_estimators': 1}},
            '2': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}},
            '3': {'dataset': 'bbb', 'classifier': {'n_estimators': 1}},
            '4': {'dataset': 'bbb', 'classifier': {'n_estimators': 2}},
            '5': {'inner_experiment': {'dataset': 'aaa', 'classifier': {'n_estimators': 1}}},
            '6': {'inner_experiment': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}}},
            '7': {'inner_experiment': {'dataset': 'bbb', 'classifier': {'n_estimators': 1}}},
            '8': {'inner_experiment': {'dataset': 'bbb', 'classifier': {'n_estimators': 2}}},
        },
    )


def repeated_dependency_evaluation(context: dict[str, Any]) -> Evaluation:
    """Ensure we correctly handle repeated references to the same
    dependency task."""
    def experiment_factory():
        return ClassifierExperiment(
            classifier_task=ClassifierTask(n_estimators=2),
            dataset_key=list(context['DATASETS'].keys())[0],
        )
    experiment = experiment_factory()
    return Evaluation(
        task=ExperimentEvaluationTask(
            experiments=[
                # Repeating the same identity.
                experiment,
                experiment,
                # Additional duplicate with different identity.
                experiment_factory(),
            ],
        ),
        expected_result={
            '1': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}},
            '2': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}},
            '3': {'dataset': 'aaa', 'classifier': {'n_estimators': 2}},
        },
    )


@pytest.fixture
def evaluations(context: dict[str, Any]) -> dict[str, Evaluation]:
    return {
        'basic': basic_evaluation(context),
        'repeated_dependency': repeated_dependency_evaluation(context),
    }


active_evaluation_keys = ['basic', 'repeated_dependency']


class TestE2E:

    @pytest.mark.parametrize('max_workers', [1, 4, None])
    @pytest.mark.parametrize('runner_backend', ['serial', 'fork', 'spawn', 'thread'])
    @pytest.mark.parametrize('evaluation_key', active_evaluation_keys)
    def test_e2e(self, max_workers: int, runner_backend: str, evaluation_key: str, context: dict[str, Any], evaluations: dict[str, Evaluation]) -> None:
        evaluation = evaluations[evaluation_key]

        # macOS and Windows don't support fork, so test graceful failure:
        if runner_backend == 'fork' and platform.system() in {'Darwin', 'Windows'}:
            lab = labtech.Lab(
                storage=None,
                context=context,
                max_workers=max_workers,
                runner_backend=runner_backend,
            )
            with pytest.raises(RunnerError, match="The 'fork' start method for processes is not supported by your operating system."):
                lab.run_task(evaluation['task'])
            return

        with TemporaryDirectory() as storage_dir:
            lab = labtech.Lab(
                storage=storage_dir,
                context=context,
                max_workers=max_workers,
                runner_backend=runner_backend,
            )
            evaluation_result = lab.run_task(evaluation['task'])
            assert evaluation_result == evaluation['expected_result']

            cached_tasks = lab.cached_tasks([type(evaluation['task'])])
            print(cached_tasks)
            print([evaluation['task']])
            assert cached_tasks == [evaluation['task']]

            cached_result = lab.run_task(cached_tasks[0])
            assert cached_result == evaluation['expected_result']

class TestE2ERay:

    def setup_method(self, method):
        ray.init(
            # See: https://docs.ray.io/en/latest/ray-contribute/testing-tips.html#tips-for-testing-ray-programs
            num_cpus=2,
            # Ensure test_e2e.py is visible to Ray
            runtime_env={
                'working_dir': 'tests/integration/',
            },
        )

    def teardown_method(self, method):
        ray.shutdown()

    @pytest.mark.parametrize('evaluation_key', active_evaluation_keys)
    def test_e2e_ray(self, evaluation_key: str, context: dict[str, Any], evaluations: dict[str, Evaluation]) -> None:
        evaluation = evaluations[evaluation_key]
        with TemporaryDirectory() as storage_dir:
            lab = labtech.Lab(
                storage=storage_dir,
                context=context,
                max_workers=None,
                runner_backend=RayRunnerBackend(),
            )
            evaluation_result = lab.run_task(evaluation['task'])
            assert evaluation_result == evaluation['expected_result']

            cached_tasks = lab.cached_tasks([type(evaluation['task'])])
            assert cached_tasks == [evaluation['task']]

            cached_result = lab.run_task(cached_tasks[0])
            assert cached_result == evaluation['expected_result']
