## Labtech Tutorial

The following tutorial presents a full example of using labtech to
easily add parallelism and caching to machine learning experiments.

You can also run this tutorial as an ([interactive notebook](https://mybinder.org/v2/gh/ben-denham/labtech/main?filepath=examples/tutorial.ipynb)).

Firstly, we will install and import `labtech` along with some other
dependencies we will use in this tutorial:

``` {.code}
%pip install labtech mlflow scikit-learn
```

### Running tasks in parallel

TODO

``` {.python .code}
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

digits_X, digits_y = datasets.load_digits(return_X_y=True)
digits_X = StandardScaler().fit_transform(digits_X)

clf = LogisticRegression(random_state=1)
clf.fit(digits_X, digits_y)
# Note: Normally we would want to test predictions on a separate test set,
# but we will work with a training set only for simplicity in this tutorial.
prob_y = clf.predict_proba(digits_X)

print(log_loss(digits_y, prob_y))
```

TODO

``` {.python .code}
import labtech

@labtech.task
class ClassifierExperiment:
    random_state: int

    def run(self):
        clf = LogisticRegression(random_state=self.random_state)
        clf.fit(digits_X, digits_y)
        return clf.predict_proba(digits_X)


classifier_experiments = [
    ClassifierExperiment(
        random_state=random_state,
    )
    for random_state in range(10)
]
```

TODO

``` {.python .code}
lab = labtech.Lab(
    storage='classification_lab_1',
    notebook=True,
)

results = lab.run_tasks(classifier_experiments)
print({
    experiment: log_loss(digits_y, prob_y)
    for experiment, prob_y in results.items()
})
```

### Dependent tasks: concurrency and re-using cached results

TODO

``` {.python .code}
lab.run_tasks(classifier_experiments)
```

```
lab.cached_tasks([ClassifierExperiment])
```

TODO

``` {.python .code}
@labtech.task
class MinMaxProbabilityExperiment:
    classifier_experiment: ClassifierExperiment

    def run(self):
        prob_y = self.classifier_experiment.result
        # Replace the maximum probability in each row with 1,
        # and replace all other probabilities with 0.
        min_max_prob_y = np.zeros(prob_y.shape)
        min_max_prob_y[np.arange(len(prob_y), prob_y.argmax(axis=1)] = 1
        return min_max_prob_y
```

TODO

``` {.python .code}
min_max_prob_experiments = [
    MinMaxProbabilityExperiment(
        classifier_experiment=classifier_experiment,
    )
    for classifier_experiment in classifier_experiments
]

results = lab.run_tasks(min_max_prob_experiments)
print({
    experiment: log_loss(digits_y, prob_y)
    for experiment, prob_y in results.items()
})
```

### Parameterising tasks with complex objects

TODO

``` {.python .code}
from typing import Protocol

from sklearn.base import clone, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


class ClassifierTask(Protocol):

    def run(self) -> ClassifierMixin:
        pass


# Constructing a classifier object is inexpensive, so we don't need to
# cache the result
@labtech.task(cache=None)
class LRClassifierTask:
    random_state: int

    def run(self) -> ClassifierMixin:
        return LogisticRegression(
            random_state=self.random_state,
        )


@labtech.task(cache=None)
class NBClassifierTask:

    def run(self) -> ClassifierMixin:
        return GaussianNB()


@labtech.task
class ClassifierExperiment:
    classifier_task: ClassifierTask

    def run(self):
        clf = clone(self.classifier_task.result)
        clf.fit(digits_X, digits_y)
        probs = clf.predict_proba(digits_X)
        return probs
```

TODO

``` {.python .code}
lr_classifier_tasks = [
    LRClassifierTask(
        random_state=random_state,
    )
    for random_state in range(10)
]
classifier_experiments = [
    ClassifierExperiment(
        classifier_task=classifier_task,
    )
    for classifier_task in [
        NBClassifierTask(),
        *lr_classifier_tasks,
    ]
]

lab = labtech.Lab(
    storage='classification_lab_2',
    notebook=True,
)

results = lab.run_tasks(classifier_experiments)
print({
    experiment: log_loss(digits_y, prob_y)
    for experiment, prob_y in results.items()
})
```

TODO

### Providing large objects as context

``` {.python .code}
iris_X, iris_y = datasets.load_iris(return_X_y=True)
iris_X = StandardScaler().fit_transform(iris_X)

DATASETS = {
    'digits': {'X': digits_X, 'y': digits_y},
    'iris': {'X': iris_X, 'y': iris_y},
}


@labtech.task
class ClassifierExperiment:
    classifier_task: ClassifierTask
    dataset_key: str

    def run(self):
        dataset = self.context['DATASETS'][self.dataset_key]
        X, y = dataset['X'], dataset['y']

        clf = clone(self.classifier_task.result)
        clf.fit(X, y)
        return clf.predict_proba(X)
```

TODO

``` {.python .code}
classifier_experiments = [
    ClassifierExperiment(
        classifier_task=classifier_task,
        dataset_key=dataset_key,
    )
    # By including multiple for clauses, we will produce a ClassifierExperiment
    # for every combination of dataset_key and classifier_task
    for dataset_key in datasets.keys()
    for classifier_task in [NBClassifierTask(), *lr_classifier_tasks]
]

lab = labtech.Lab(
    storage='classification_lab_3',
    notebook=True,
    context={
        'DATASETS': DATASETS,
    },
)

results = lab.run_tasks(classifier_experiments)
print({
    experiment: log_loss(digits_y, prob_y)
    for experiment, prob_y in results.items()
})
```

### Bringing it all together and aggregating results

TODO

``` {.python .code}
from typing import Protocol

import labtech
from sklearn.base import clone, ClassifierMixin


# === Prepare Datasets ===

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

digits_X, digits_y = datasets.load_digits(return_X_y=True)
digits_X = StandardScaler().fit_transform(digits_X)

iris_X, iris_y = datasets.load_iris(return_X_y=True)
iris_X = StandardScaler().fit_transform(iris_X)

DATASETS = {
    'digits': {'X': digits_X, 'y': digits_y},
    'iris': {'X': iris_X, 'y': iris_y},
}


# === Classifier Tasks ===

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ClassifierTask(Protocol):

    def run(self) -> ClassifierMixin:
        pass


# Constructing a classifier object is inexpensive, so we don't need to
# cache the result
@labtech.task(cache=None)
class LRClassifierTask:
    random_state: int

    def run(self) -> ClassifierMixin:
        return LogisticRegression(
            random_state=self.random_state,
        )


@labtech.task(cache=None)
class NBClassifierTask:

    def run(self) -> ClassifierMixin:
        return GaussianNB()


# === Experiment Tasks ===

class ExperimentTask(Protocol):

    def run(self) -> np.ndarray:
        pass


@labtech.task(mlflow_run=True)
class ClassifierExperiment:
    classifier_task: ClassifierTask
    dataset_key: str

    def run(self) -> np.ndarray:
        dataset = self.context['DATASETS'][self.dataset_key]
        X, y = dataset['X'], dataset['y']

        clf = clone(self.classifier_task.result)
        clf.fit(X, y)
        return clf.predict_proba(X)


@labtech.task(mlflow_run=True)
class MinMaxProbabilityExperiment:
    experiment: ExperimentTask

    def run(self) -> np.ndarray:
        prob_y = self.experiment.result
        # Replace the maximum probability in each row with 1,
        # and replace all other probabilities with 0.
        min_max_prob_y = np.zeros(prob_y.shape)
        min_max_prob_y[np.arange(len(prob_y), prob_y.argmax(axis=1)] = 1
        return min_max_prob_y


# === Results Aggregation ===

from sklearn.metrics import log_loss

@labtech.task
class ExperimentEvaluationTask:
    experiments: list[ExperimentTask]

    def run(self):
        return {
            experiment: log_loss(digits_y, experiment.result)
            for experiment in self.experiments
        }


# === Task Construction ===

lr_classifier_tasks = [
    LRClassifierTask(
        random_state=random_state,
    )
    for random_state in range(10)
]

classifier_experiments = [
    ClassifierExperiment(
        classifier_task=classifier_task,
        dataset_key=dataset_key,
    )
    for dataset_key in datasets.keys()
    for classifier_task in [NBClassifierTask(), *lr_classifier_tasks]
]

min_max_prob_experiments = [
    MinMaxProbabilityExperiment(
        classifier_experiment=classifier_experiment,
    )
    for classifier_experiment in classifier_experiments
]

evaluation_task = ExperimentEvaluationTask(
    experiments=[
        *classifier_experiments,
        *min_max_prob_experiments,
    ]
)


# === Task Execution ===

import mlflow

mlflow.set_experiment('example_labtech_experiment')
lab = labtech.Lab(
    storage='classification_lab_final',
    notebook=True,
    context={
        'DATASETS': DATASETS,
    },
)

evaluation_result = lab.run_task(evaluation_task)
print(evaluation_result)
```

### Next Steps

TODO
