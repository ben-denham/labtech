<div align="center">

<img alt="Labtech logo - FIFO the Labrador in a lab coat and goggles working on a laptop with the labtech 'lt' icon." src="https://ben-denham.github.io/labtech/images/fifo.svg">

<h1>labtech</h1>

<a href="">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/labtech">
</a>

<p>
    <a href="https://github.com/ben-denham/labtech">GitHub</a> - <a href="https://ben-denham.github.io/labtech">Documentation</a>
</p>

</div>

Labtech makes it easy to define multi-step experiment pipelines and
run them with maximal parallelism and result caching:

* **Defining tasks is simple**; write a class with a single `run()`
  method and parameters as dataclass-style attributes.
* **Flexible experiment configuration**; simply create task objects
  for all of your parameter permutations.
* **Handles pipelines of tasks**; any task parameter that is itself a
  task will be executed first and make its result available to its
  dependent task(s).
* **Implicit parallelism**; Labtech resolves task dependencies and
  runs tasks in sub-processes with as much parallelism as possible.
* **Implicit caching and loading of task results**; configurable and
  extensible options for how and where task results are cached.
* **Integration with [mlflow](https://mlflow.org/)**; Automatically
  log task runs to mlflow with all of their parameters.
* **Easily scale to a multi-machine cluster**; Built-in support for
  running tasks across an easy-to-setup [Ray](https://www.ray.io/)
  cluster.

To learn more about how Labtech can speed up your experiments, check
out the Kiwi Pycon presentation:

<a href="https://youtube.com/watch?v=LX5kFOVLwaQ" title="Labtech: Concurrency and caching in Python for busy scientists — Ben Denham">
    <img src="https://img.youtube.com/vi/LX5kFOVLwaQ/0.jpg" alt="Kiwi Pycon conference recording of "Labtech: Concurrency and caching in Python for busy scientists"">
</a>

## Installation

```
pip install labtech
```


## Usage

<!-- N.B. keep this code in-sync with tests/integration/readme/usage.py -->
```python
from time import sleep

import labtech

# Decorate your task class with @labtech.task:
@labtech.task
class Experiment:
    # Each Experiment task instance will take `base` and `power` parameters:
    base: int
    power: int

    def run(self) -> int:
        # Define the task's run() method to return the result of the experiment:
        labtech.logger.info(f'Raising {self.base} to the power of {self.power}')
        sleep(1)
        return self.base ** self.power

def main():
    # Configure Experiment parameter permutations
    experiments = [
        Experiment(
            base=base,
            power=power,
        )
        for base in range(5)
        for power in range(5)
    ]

    # Configure a Lab to run the experiments:
    lab = labtech.Lab(
        # Specify a directory to cache results in (running the experiments a second
        # time will just load results from the cache!):
        storage='demo_lab',
        # Control the degree of parallelism:
        max_workers=5,
    )

    # Run the experiments!
    results = lab.run_tasks(experiments)
    print([results[experiment] for experiment in experiments])

if __name__ == '__main__':
    main()
```

![Animated GIF of labtech demo on the command-line](https://ben-denham.github.io/labtech/images/labtech-demo.gif)

Labtech can also produce graphical progress bars in
[Jupyter](https://jupyter.org/) notebooks:

![Animated GIF of labtech demo in Jupyter](https://ben-denham.github.io/labtech/images/labtech-demo-jupyter.gif)

Tasks parameters can be any of the following types:

* Simple scalar types: `str`, `bool`, `float`, `int`, `None`
* Collections of any of these types: `list`, `tuple`, `dict`, `Enum`
* Task types: A task parameter is a "nested task" that will be
  executed before its parent so that it may make use of the nested
  result.

Here's an example of defining a single long-running task to produce a
result for a large number of dependent tasks:

<!-- N.B. keep this code in-sync with tests/integration/readme/dependents_and_mermaid.py -->
```python
from time import sleep

import labtech

@labtech.task
class SlowTask:
    base: int

    def run(self) -> int:
        sleep(5)
        return self.base ** 2

@labtech.task
class DependentTask:
    slow_task: SlowTask
    multiplier: int

    def run(self) -> int:
        return self.multiplier * self.slow_task.result

def main():
    some_slow_task = SlowTask(base=42)
    dependent_tasks = [
        DependentTask(
            slow_task=some_slow_task,
            multiplier=multiplier,
        )
        for multiplier in range(10)
    ]

    lab = labtech.Lab(storage='demo_lab')
    results = lab.run_tasks(dependent_tasks)
    print([results[task] for task in dependent_tasks])

if __name__ == '__main__':
    main()
```

Labtech can even generate a [Mermaid diagram](https://mermaid.js.org/syntax/classDiagram.html)
to visualise your tasks:

<!-- N.B. keep this code in-sync with tests/integration/readme/dependents_and_mermaid.py -->
```python
from labtech.diagram import display_task_diagram

some_slow_task = SlowTask(base=42)
dependent_tasks = [
    DependentTask(
        slow_task=some_slow_task,
        multiplier=multiplier,
    )
    for multiplier in range(10)
]

display_task_diagram(dependent_tasks)
```

```mermaid
classDiagram
    direction BT

    class DependentTask
    DependentTask : SlowTask slow_task
    DependentTask : int multiplier
    DependentTask : run() int

    class SlowTask
    SlowTask : int base
    SlowTask : run() int


    DependentTask <-- SlowTask: slow_task
```

To learn more, dive into the following resources:

* [The labtech tutorial](https://ben-denham.github.io/labtech/tutorial) ([as an interactive notebook](https://mybinder.org/v2/gh/ben-denham/labtech/main?filepath=examples/tutorial.ipynb))
* [Cookbook of common patterns](https://ben-denham.github.io/labtech/cookbook) ([as an interactive notebook](https://mybinder.org/v2/gh/ben-denham/labtech/main?filepath=examples/cookbook.ipynb))
* [API reference for Labs and Tasks](https://ben-denham.github.io/labtech/core)
* [More options for cache formats and storage providers](https://ben-denham.github.io/labtech/caching)
* [Option for customising task execution backends](https://ben-denham.github.io/labtech/runners)
* [Diagramming tools](https://ben-denham.github.io/labtech/diagram)
* [Distributing across multiple machines](https://ben-denham.github.io/labtech/distributed)
* [More examples](https://github.com/ben-denham/labtech/tree/main/examples)


## Mypy Plugin

For [mypy](https://mypy-lang.org/) type-checking of classes decorated
with `labtech.task`, simply enable the labtech mypy plugin in your
`mypy.ini` file:

```INI
[mypy]
plugins = labtech.mypy_plugin
```

## Contributing

* Install uv dependencies with `make deps`
* Run linting, mypy, and tests with `make check`
* Documentation:
    * Run local server: `make docs-serve`
    * Build docs: `make docs-build`
    * Deploy docs to GitHub Pages: `make docs-github`
    * Docstring style follows the [Google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
