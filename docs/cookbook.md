## Cookbook

The following cookbook presents labtech patterns for common use cases.

You can also run this cookbook as an ([interactive notebook](https://mybinder.org/v2/gh/ben-denham/labtech/main?filepath=examples/cookbook.ipynb)).

``` {.code}
%pip install labtech mlflow scikit-learn
```

``` {.python .code}
import labtech

import numpy as np
from sklearn import datasets
from sklearn.base import clone, ClassifierMixin
from sklearn.preprocessing import StandardScaler

# Prepare a dataset for examples
digits_X, digits_y = datasets.load_digits(return_X_y=True)
digits_X = StandardScaler().fit_transform(digits_X)
```

### How can I print log messages from my task?

Using `labtech.logger` (a standard [Python logger
object](https://docs.python.org/3/library/logging.html#logger-objects))
is the recommended approach for logging from a task, but all output
that is sent to `STDOUT` (e.g. calls to `print()`) or `STDERR` (e.g.
uncaught exceptions) will also be captured and logged:

``` {.python .code}
@labtech.task
class PrintingExperiment:
    seed: int

    def run(self):
        labtech.logger.warning(f'Warning, the seed is: {self.seed}')
        print(f'The seed is: {self.seed}')
        return self.seed * self.seed


experiments = [
    PrintingExperiment(
        seed=seed
    )
    for seed in range(5)
]
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
results = lab.run_tasks(experiments)
```


### How do I specify a complex object, like a model or dataset, as a task parameter?

Because labtech needs to be able to reconstitute `Task` objects from
caches, task parameters can only be:

* Simple scalar types: `str`, `bool`, `float`, `int`, `None`
* Any member of an `Enum` type.
* Task types: A task parameter is a "nested task" that will be executed
  before its parent so that it may make use of the nested result.
* Collections of any of these types: `list`, `tuple`,
  `dict`, [`frozendict`](https://pypi.org/project/frozendict/)
  * Note: Mutable `list` and `dict` collections will be converted to
    immutable `tuple` and [`frozendict`](https://pypi.org/project/frozendict/)
    collections.

The are three primary patterns you can use to provide a more complex
object as a parameter to a task:

* Constructing the object in a dependent task
* Passing the object in an `Enum` parameter
* Passing the object in the lab context

#### Constructing objects in dependent tasks

If your object can be constructed from its own set of parameters, then
you can use a dependent task as a "factory" to construct your object.

For example, you could define a task type to construct a machine
learning model (like `LRClassifierTask` below), and then make a task
of that type a parameter for your primary experiment task:

``` {.python .code}
from sklearn.linear_model import LogisticRegression


# Constructing a classifier object is inexpensive, so we don't need to cache the result
@labtech.task(cache=None)
class LRClassifierTask:
    random_state: int

    def run(self) -> ClassifierMixin:
        return LogisticRegression(
            random_state=self.random_state,
        )


@labtech.task
class ClassifierExperiment:
    classifier_task: LRClassifierTask

    def run(self) -> np.ndarray:
        # Because the classifier task result may be shared between experiments,
        # we clone it before fitting.
        clf = clone(self.classifier_task.result)
        clf.fit(digits_X, digits_y)
        return clf.predict_proba(digits_X)


experiment = ClassifierExperiment(
    classifier_task=LRClassifierTask(random_state=42),
)
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
results = lab.run_tasks([experiment])
```

We can extend this example with additional task types to cater for
different types of classifiers with a Protocol that defines their
common result type:

``` {.python .code}
from typing import Protocol

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


class ClassifierTask(Protocol):

    def run(self) -> ClassifierMixin:
        pass


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

    def run(self) -> np.ndarray:
        # Because the classifier task result may be shared between experiments,
        # we clone it before fitting.
        clf = clone(self.classifier_task.result)
        clf.fit(digits_X, digits_y)
        return clf.predict_proba(digits_X)


classifier_tasks = [
    LRClassifierTask(random_state=42),
    NBClassifierTask(),
]
experiments = [
    ClassifierExperiment(classifier_task=classifier_task)
    for classifier_task in classifier_tasks
]
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
results = lab.run_tasks(experiments)
```

#### Passing objects in `Enum` parameters

For simple object parameters that have a fixed set of known values, an
`Enum` of possible values can be used to provide parameter values.

The following example shows how an `Enum` of functions can be used for
a parameter to specify the operation that an experiment performs:

> Note: Because parameter values will be
> [pickled](https://docs.python.org/3/library/pickle.html) when they
> are copied to parallel task sub-processes, the type used in a
> parameter `Enum` must support equality between identical (but
> distinct) object instances.

``` {.python .code}
from enum import Enum
from datetime import datetime


# The custom class we want to provide objects of as parameters.
class Dataset:

    def __init__(self, key):
        self.key = key
        self.data = datasets.fetch_openml(key, parser='auto').data

    def __eq__(self, other):
        # Support equality check to allow pickling of Enum values
        if type(self) != type(other):
            return False
        return self.key == other.key


class DatasetOption(Enum):
    TIC_TAC_TOE=Dataset('tic-tac-toe')
    EEG_EYE_STATE=Dataset('eeg-eye-state')


@labtech.task
class DatasetExperiment:
    dataset: DatasetOption

    def run(self):
        dataset = self.dataset.value
        return dataset.data.shape


experiments = [
    DatasetExperiment(
        dataset=dataset
    )
    for dataset in DatasetOption
]
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
results = lab.run_tasks(experiments)
```

#### Passing objects in the lab context

If an object cannot be conveniently defined in an `Enum` (such as
types like Numpy arrays or Pandas DataFrames that cannot be directly
specified as an `Enum` value, or large values that cannot be loaded
into memory all at once), then the lab context can be used to pass the
object to a task.

> Note: Because values provided in the lab context are not cached,
> they should be kept constant between runs or should not affect task
> results (e.g. parallel worker counts, log levels).

The following example demonstrates specifying a `dataset_key`
parameter to a task that is used to look up a dataset from the lab
context:

``` {.python .code}
DATASETS = {
    'zeros': np.zeros((50, 10)),
    'ones': np.ones((50, 10)),
}


@labtech.task
class SumExperiment:
    dataset_key: str

    def run(self):
        dataset = self.context['DATASETS'][self.dataset_key]
        return np.sum(dataset)


experiments = [
    SumExperiment(
        dataset_key=dataset_key
    )
    for dataset_key in DATASETS.keys()
]
lab = labtech.Lab(
    storage=None,
    notebook=True,
    context={
        'DATASETS': DATASETS,
    },
)
results = lab.run_tasks(experiments)
```


### How can I control multi-processing myself within a task?

By default, Labtech executes tasks in parallel on all available CPU
cores. However, if you wish to control multi-processing yourself

The following example uses `max_parallel` to allow only one
`CVExperiment` to be executed at a time, and then performs
cross-validation within the task using a number of workers specified
in the lab context as `within_task_workers`:

``` {.python .code}
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


@labtech.task
class CVExperiment:
    cv_folds: int

    def run(self):
        clf = GaussianNB()
        return cross_val_score(
            clf,
            digits_X,
            digits_y,
            cv=self.cv_folds,
            n_jobs=self.context['within_task_workers'],
        )


experiments = [
    CVExperiment(
        cv_folds=cv_folds
    )
    for cv_folds in [5, 10]
]
lab = labtech.Lab(
    storage=None,
    notebook=True,
    max_workers=1,
    context={
        'within_task_workers': 4,
    },
)
results = lab.run_tasks(experiments)
```

> Note: Instead of limiting parallelism across all tasks with
> `max_workers` when constructing a `labtech.Lab`, you can limit
> parallelism for a single task type by specifying `max_parallel` in
> the `@labtech.task` decorator. However, the `joblib` library used by
> `sklearn` does not behave correctly when run from within a
> sub-process, and only setting `max_workers=1` ensures all tasks run
> inside the main process.


### What happens if I move or change the definition of a task?

### How can I cache task results in a format other than pickle?

### How can I cache task results somewhere other than my filesystem?

### Loading lots of cached results is slow, how can I make it faster?

### How can I use labtech with mlflow?

### How can I construct a multi-step experiment pipeline?
