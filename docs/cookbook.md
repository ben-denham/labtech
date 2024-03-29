## Labtech Cookbook

The following cookbook presents labtech patterns for common use cases.

You can also run this cookbook as an [interactive notebook](https://mybinder.org/v2/gh/ben-denham/labtech/main?filepath=examples/cookbook.ipynb).

``` {.code}
%pip install labtech fsspec mlflow pandas scikit-learn setuptools
```

``` {.code}
!mkdir storage
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


# Constructing a classifier object is inexpensive, so we don't need to
# cache the result
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
different types of classifiers with a
[Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
that defines their common result type:

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
specified as an `Enum` value, or large values that cannot all be
loaded into memory every time the `Enum` is loaded), then the lab
context can be used to pass the object to a task.

> Warning: Because values provided in the lab context are not cached,
> they should be kept constant between runs or should not affect task
> results (e.g. parallel worker counts, log levels). If changing
> context values cause task results to change, then cached results may
> no longer be valid.

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
cores. However, you can control multi-processing yourself by disabling
task parallelism and performing your own parallelism within a task's
`run()` method.

The following example uses `max_parallel` to allow only one
`CVExperiment` to be executed at a time, and then performs
cross-validation within the task using a number of workers specified
in the lab context as `within_task_workers`:

``` {.python .code}
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


@labtech.task(max_parallel=1)
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
    context={
        'within_task_workers': 4,
    },
)
results = lab.run_tasks(experiments)
```

> Note: Instead of limiting parallelism for a single task type by
> specifying `max_parallel` in the `@labtech.task` decorator, you can
> limit parallelism across all tasks with `max_workers` when
> constructing a `labtech.Lab`.

> Note: The `joblib` library used by `sklearn` does not behave
> correctly when run from within a task sub-process, but setting
> `max_parallel=1` or `max_workers=1` ensures tasks are run inside the
> main process.

### How can I make labtech continue executing tasks even when one or more fail?

Labtech's default behaviour is to stop executing any new tasks as soon
as any individual task fails. However, when executing tasks over a
long period of time (e.g. a large number of tasks, or even a few long
running tasks), it is sometimes helpful to have labtech continue to
execute tasks even if one or more fail.

If you set `continue_on_failure=True` when creating your lab,
exceptions raised during the execution of a task will be logged, but
the execution of other tasks will continue:

``` {.python .code}
lab = labtech.Lab(
    storage=None,
    continue_on_failure=True,
)
```

### What happens to my cached results if I change or move the definition of a task?

A task's cache will store all details necessary to reinstantiate the
task object, including the qualified name of the task's class and all
of the task's parameters. Because of this, it is best not to change
the parameters and location of a task's definition once you are
seriously relying on cached results.

If you need to add a new parameter or behaviour to an existing task
type for which you have previously cached results, consider defining a
sub-class for that extension so that you can continue using caches for
the base class:

``` {.python .code}
@labtech.task
class Experiment:
    seed: int

    def run(self):
        return self.seed * self.seed


@labtech.task
class ExtendedExperiment(Experiment):
    multiplier: int

    def run(self):
        base_result = super().run()
        return base_result * self.multiplier
```

### How can I find what results I have cached?

You can use the `cached_task()` method of a `Lab` instance to retrieve
all cached task instances for a list of task types. You can then "run"
the tasks to load their cached results:

``` {.python .code}
cached_cvexperiment_tasks = lab.cached_tasks([CVExperiment])
results = lab.run_tasks(cached_cvexperiment_tasks)
```

### How can I clear cached results?

You can clear the cache for a list of tasks using the
`uncache_tasks()` method of a `Lab` instance:

``` {.python .code}
lab.uncache_tasks(cached_cvexperiment_tasks)
```

You can also ignore all previously cached results when running a list
of tasks by passing the `bust_cache` option to `run_tasks()`:

``` {.python .code}
lab.run_tasks(cached_cvexperiment_tasks, bust_cache=True)
```

### How can I cache task results in a format other than pickle?

You can define your own cache type to support storing cached results
in a format other than pickle. To do so, you must define a class that
extends `labtech.cache.BaseCache` and defines `KEY_PREFIX`,
`save_result()`, and `load_result()`. You can then configure any task
type by passing an instance of your new cache type for the `cache`
option of the `@labtech.task` decorator.

The following example demonstrates defining and using a custom cache
type to store Pandas DataFrames as parquet files:

``` {.python .code}
from labtech.cache import BaseCache
from labtech.types import Task, ResultT
from labtech.storage import Storage
from typing import Any
import pandas as pd


class ParquetCache(BaseCache):
    """Caches a Pandas DataFrame result as a parquet file."""
    KEY_PREFIX = 'parquet__'

    def save_result(self, storage: Storage, task: Task[ResultT], result: ResultT):
        if not isinstance(result, pd.DataFrame):
            raise ValueError('ParquetCache can only cache DataFrames')
        with storage.file_handle(task.cache_key, 'result.parquet', mode='wb') as data_file:
            result.to_parquet(data_file)

    def load_result(self, storage: Storage, task: Task[ResultT]) -> ResultT:
        with storage.file_handle(task.cache_key, 'result.parquet', mode='rb') as data_file:
            return pd.read_parquet(data_file)


@labtech.task(cache=ParquetCache())
class TabularTask:

    def run(self):
        return pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1, 4, 9],
        })


lab = labtech.Lab(
    storage='storage/parquet_example',
    notebook=True,
)
lab.run_tasks([TabularTask()])
```

### How can I cache task results somewhere other than my filesystem?

You can cache results in a location other than the local filesystem by
defining your own storage type that extends `labtech.storage.Storage`
and defines `find_keys()`, `exists()`, `file_handle()` and `delete()`.
You can then pass an instance of your new storage backend for the
`storage` option when constructing a `Lab` instance.

The following example demonstrates constructing a storage backend to
interface to the `LocalFileSystem` provided by the
[`fsspec`](https://filesystem-spec.readthedocs.io) library. This
example could be adapted for other `fsspec` implementations, such as
cloud storage providers like [Amazon S3](https://s3fs.readthedocs.io/en/latest/)
and [Azure Blob Storage](https://github.com/fsspec/adlfs):

``` {.python .code}
from labtech.storage import Storage
from typing import IO, Sequence
from pathlib import Path
from fsspec.implementations.local import LocalFileSystem


class LocalFsspecStorage(Storage):
    """Store results in the local file filesystem."""

    def __init__(self, storage_dir):
        self.storage_dir = Path(storage_dir).resolve()
        fs = self._fs('w')
        fs.mkdirs(self.storage_dir, exist_ok=True)

    def _fs(self, mode):
        return LocalFileSystem()

    def _key_to_path(self, key):
        return self.storage_dir / key

    def find_keys(self) -> Sequence[str]:
        return [str(Path(entry).relative_to(self.storage_dir)) for entry in self._fs('r').ls(self.storage_dir)]

    def exists(self, key: str) -> bool:
        return self._fs('r').exists(self._key_to_path(key))

    def file_handle(self, key: str, filename: str, *, mode: str = 'r') -> IO:
        fs_mode = 'w' if 'w' in mode else 'r'
        fs = self._fs(fs_mode)
        key_path = self._key_to_path(key)
        fs.mkdirs(key_path, exist_ok=True)
        file_path = (key_path / filename).resolve()
        if file_path.parent != key_path:
            raise ValueError((f"Filename '{filename}' should only reference a directory directly "
                              f"under the storage key directory '{key_path}'"))
        return fs.open(file_path, mode)

    def delete(self, key: str):
        fs = self._fs('w')
        path = self._key_to_path(key)
        if fs.exists(path):
            fs.rm(path, recursive=True)


@labtech.task
class Experiment:
    seed: int

    def run(self):
        return self.seed * self.seed


experiments = [
    Experiment(
        seed=seed
    )
    for seed in range(100)
]
lab = labtech.Lab(
    storage=LocalFsspecStorage('storage/fsspec_example'),
    notebook=True,
)
results = lab.run_tasks(experiments)
```

<!--

# Simple testing of LocalFsspecStorage
storage = LocalFsspecStorage('storage/lab_example')
key = 'key1'
print(storage.exists(key))
storage.delete(key)
with storage.file_handle(key, 'example.txt', mode='w') as file:
    file.write('content')
with storage.file_handle(key, 'example.txt', mode='r') as file:
    print(file.read())
print(storage.find_keys())
print(storage.exists(key))
storage.delete(key)
print(storage.find_keys())

-->

### Loading lots of cached results is slow, how can I make it faster?

If you have a large number of tasks, you may find that the overhead of
loading each individual task result from the cache is unacceptably
slow when you need to frequently reload previous results for analysis.

In such cases, you may find it helpful to create a final task that
depends on all of your individual tasks and aggregates all of their
results into a single cached result. Note that this final result cache
will need to be rebuilt whenever any of its dependent tasks changes or
new dependent tasks are added. Furthermore, this approach will require
additional storage for the final cache in addition to the individual
result caches.

The following example demonstrates defining and using an
`AggregationTask` to aggregate the results from many individual tasks
to create an aggregated cache that can be loaded more efficiently:

``` {.python .code}
from labtech.types import Task

@labtech.task
class Experiment:
    seed: int

    def run(self):
        return self.seed * self.seed


@labtech.task
class AggregationTask:
    sub_tasks: list[Task]

    def run(self):
        return [
            sub_task.result
            for sub_task in self.sub_tasks
        ]


experiments = [
    Experiment(
        seed=seed
    )
    for seed in range(1000)
]
aggregation_task = AggregationTask(
    sub_tasks=experiments,
)
lab = labtech.Lab(
    storage='storage/aggregation_lab',
    notebook=True,
)
result = lab.run_task(aggregation_task)
```

### How can I see when a task was run and how long it took to execute?

Once a task has been executed (or loaded from cache), you can see when
it was originally executed and how long it took to execute from the
task's `.result_meta` attribute:

``` {.python .code}
print(f'The task was executed at: {aggregation_task.result_meta.start}')
print(f'The task execution took: {aggregation_task.result_meta.duration}')
```

### How can I access the results of intermediate/dependency tasks?

To conserve memory, labtech's default behaviour is to unload the
results of intermediate/dependency tasks once their directly dependent
tasks have finished executing.

A simple approach to access the results of an intermediate task may
simply be to include it's results as part of the result of the task
that depends on it - that way you only need to look at the results of
the final task(s).

Another approach is to include all of the intermediate tasks for which
you wish to access the results for in the call to `run_tasks()`:

``` {.python .code}
experiments = [
    Experiment(
        seed=seed
    )
    for seed in range(10)
]
aggregation_task = AggregationTask(
    sub_tasks=experiments,
)
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
results = lab.run_tasks([
    aggregation_task,
    # Include intermediate tasks to access their results
    *experiments,
])
print([
    results[experiment]
    for experiment in experiments
])
```

You can also configure labtech to not remove intermediate results from
memory by setting `keep_nested_results=True` when calling
`run_tasks()`. Intermediate results can then be accessed from the
`.result` attribute of all task objects. However, only results that
needed to be executed or loaded from cache in order to produce the
final result will be available, so you may need to set
`bust_cache=True` to ensure all intermediate tasks are executed:

``` {.python .code}
experiments = [
    Experiment(
        seed=seed
    )
    for seed in range(10)
]
aggregation_task = AggregationTask(
    sub_tasks=experiments,
)
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
result = lab.run_task(
    aggregation_task,
    keep_nested_results=True,
    bust_cache=True,
)
print([
    experiment.result
    for experiment in experiments
])
```

### How can I construct a multi-step experiment pipeline?

Say you want to model a multi-step experiment pipeline, where `StepA`
is run before `StepB`, which is run before `StepC`:

```
StepA -> StepB -> StepC
```

This is modeled in labtech by defining a task type for each step, and
having each step depend on the result from the previous step:

``` {.python .code}
@labtech.task
class StepA:
    seed_a: int

    def run(self):
        return self.seed_a


@labtech.task
class StepB:
    task_a: StepA
    seed_b: int

    def run(self):
        return self.task_a.result * self.seed_b


@labtech.task
class StepC:
    task_b: StepB
    seed_c: int

    def run(self):
        return self.task_b.result * self.seed_c


task_a = StepA(
    seed_a=2,
)
task_b = StepB(
    seed_b=3,
    task_a=task_a,
)
task_c = StepC(
    seed_c=5,
    task_b=task_b,
)

lab = labtech.Lab(
    storage=None,
    notebook=True,
)
result = lab.run_task(task_c)
print(result)
```


### How can I visualise my task types, including their parameters and dependencies?

`labtech.diagram.display_task_diagram()` can be used to display a
[Mermaid diagram](https://mermaid.js.org/syntax/classDiagram.html) of
task types for a given list of tasks:

``` {.python .code}
from labtech.diagram import display_task_diagram

display_task_diagram(
    [task_c],
    direction='RL',
)
```

`labtech.diagram.build_task_diagram()` can be similarly used to return
the Mermaid syntax for the diagram.


### How can I use labtech with mlflow?

If you want to log a task type as an mlflow "run", simply add
`mlflow_run=True` to the call to `@labtech.task()`, which will:

* Wrap each run of the task with `mlflow.start_run()`
* Tag the run with `labtech_task_type` equal to the task class name
* Log all task parameters with `mlflow.log_param()`

The following example demonstrates using labtech with mlflow. Note
that you can still make any configuration changes (such as
`mlflow.set_experiment()`) before the tasks are run, and you can make
additional tracking calls (such as `mlflow.log_metric()` or
`mlflow.log_model()`) in the body of your task's `run()` method:

``` {.python .code}
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@labtech.task(mlflow_run=True)
class MLRun:
    penalty_norm: str | None

    def run(self) -> np.ndarray:
        clf = LogisticRegression(penalty=self.penalty_norm)
        clf.fit(digits_X, digits_y)

        labels = clf.predict(digits_X)

        train_accuracy = accuracy_score(digits_y, labels)
        mlflow.log_metric('train_accuracy', train_accuracy)

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path='digits_model',
            input_example=digits_X,
            registered_model_name='digits-model',
        )

        return labels


runs = [
    MLRun(
        penalty_norm=penalty_norm,
    )
    for penalty_norm in [None, 'l2']
]

mlflow.set_experiment('example_labtech_experiment')
lab = labtech.Lab(
    storage=None,
    notebook=True,
)
results = lab.run_tasks(runs)
```

> Note: While the [mlflow documentation](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html#step-4-log-the-model-and-its-metadata-to-mlflow)
> recommends wrapping only your tracking code with
> `mlflow.start_run()`, labtech wraps the entire call to the `run()`
> method of your task in order to track execution times in mlflow.


### Why do I see the following error: `An attempt has been made to start a new process before the current process has finished`?

When running labtech in a Python script on Windows, macOS, or any
Python environment using the `spawn` multiprocessing start method, you
will see the following error if you do not guard your experiment and
lab creation and other non-definition code with `__name__ ==
'__main__'`:

```
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
```

To avoid this error, it is recommended that you write all of your
non-definition code for a Python script in a `main()` function, and
then guard the call to `main()` with `__name__ == '__main__'`:

``` {.python .code}
import labtech

@labtech.task
class Experiment:
    seed: int

    def run(self):
        return self.seed * self.seed

def main():
    experiments = [
        Experiment(
            seed=seed
        )
        for seed in range(1000)
    ]
    lab = labtech.Lab(
        storage='storage/guarded_lab',
        notebook=True,
    )
    result = lab.run_tasks(experiments)
    print(result)

if __name__ == '__main__':
    main()
```

For details, see [Safe importing of main module](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import).
