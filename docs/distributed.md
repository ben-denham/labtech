Labtech offers built-in support for distributing your tasks across a
multi-machine [Ray](https://www.ray.io/) cluster with the
[`RayRunnerBackend`][labtech.runners.ray.RayRunnerBackend].

* Ray manages [distributing tasks](https://docs.ray.io/en/latest/ray-core/scheduling/index.html)
  to available nodes across your cluster, and its
  [built-in object store](https://docs.ray.io/en/latest/ray-core/objects/serialization.html)
  is used to share the lab context and task results across nodes.
* Ray's [locality-aware scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/index.html#locality-aware-scheduling)
  will prefer scheduling tasks to nodes where the results of dependency
  tasks are already available.
* For task results that are [NumPy](https://numpy.org/) arrays, Ray
  uses [zero-copy deserialization](https://docs.ray.io/en/latest/ray-core/objects/serialization.html)
  to share the in-memory array between all workers on the same node.

You can also use distributed computation platforms other than Ray with
Labtech by implementing a [custom task runner
backend](./runners.md#custom-task-runner-backends), but this
documentation will focus on Labtech's built-in Ray support.


!!! tip

    Given the inherent complexities that come with managing a cluster of
    machines, you should always consider whether the scale of your tasks
    really requires it - public cloud providers offer "spot" (i.e.
    discounted cost, but evictable) virtual machines with dozens of CPU
    cores and hundreds of GB in memory for less than a dollar an hour,
    and these may be a perfect fit for easily scaling your tasks.


## Running Labtech tasks on a Ray cluster

Follow these steps to get Labtech tasks running across a Ray cluster.
Because Ray makes it easy to run a local cluster, you can even try
Labtech with Ray without setting up a multi-machine cluster.

### Installing Ray

`ray` is an optional dependency of Labtech, so you must explicitly
install it with `pip` (or your preferred Python package manager). It
is recommended that you install `ray[default]` on the machine you
intend to start the cluster from so that you can enable Ray's built-in
[dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html):

```bash
pip install "ray[default]"
```

### Using distributed storage

Because each Labtech task is responsible for caching its result to
persistent storage, you must use a storage backend that can be
accessed from any node in the cluster. For example, you could use an
[NFS share](https://en.wikipedia.org/wiki/Network_File_System) or a
cloud object storage provider (e.g. Amazon S3 or Azure Blob Storage).

To learn how to configure Labtech to use a non-local storage backend, see:
[How can I cache task results somewhere other than my filesystem?](./cookbook.md#how-can-i-cache-task-results-somewhere-other-than-my-filesystem)

In the following example, we will run a
[LocalStack](https://www.localstack.cloud/) instance to emulate an
[Amazon S3 object storage bucket](https://aws.amazon.com/s3/). For
testing, you can run your own LocalStack S3 bucket named
`labtech-dev-bucket` with [Docker Compose](https://docs.docker.com/compose/) by
creating the following `docker-compose.yml` and running
`docker compose up localstack`:

```yaml
# docker-compose.yml
services:
  localstack:
    image: localstack/localstack:4.3
    ports:
      - "127.0.0.1:4566:4566"            # LocalStack Gateway
      - "127.0.0.1:4510-4559:4510-4559"  # external services port range
    volumes:
      - "./.localstack:/var/lib/localstack"
    post_start:
      - command: awslocal s3api create-bucket --bucket labtech-dev-bucket
```

### Code example

The following code demonstrates how to configure a Lab to run tasks
across a Ray cluster:

```python
# Ray defaults to de-duplicating similar log messages. To show all log
# messages from tasks, the RAY_DEDUP_LOGS environment variable must be
# set to zero **before** importing ray and labtech.runners.ray. See:
# https://docs.ray.io/en/latest/ray-observability/user-guides/
#     configure-logging.html#log-deduplication
import os
os.environ['RAY_DEDUP_LOGS'] = '0'

import labtech
import ray
from labtech.storage import FsspecStorage
from labtech.runners.ray import RayRunnerBackend
from s3fs import S3FileSystem


# Connect to a Ray cluster:
# * If no cluster is running locally, `ray.init()` will start one.
# * If you've started a local cluster with `ray start --head --port 6379`,
#   `ray.init()` will connect to it.
#   See: https://docs.ray.io/en/latest/ray-core/starting-ray.html
# * You can specify the address of a remote cluster,
#   e.g. `ray.init(address='ray://123.45.67.89:10001')`
ray.init()


# Define a custom Storage backend for our localstack S3 bucket
# using s3fs (which implements the fsspec interface)
class S3fsStorage(FsspecStorage):

    def fs_constructor(self):
        return S3FileSystem(
            # Use localstack endpoint:
            endpoint_url='http://localhost:4566',
            key='anything',
            secret='anything',
        )


@labtech.task
class Experiment:
    seed: int

    def run(self):
        labtech.logger.info(f'Running with seed {self.seed}')
        return self.seed

experiments = [Experiment(seed=seed) for seed in range(10)]


# Configure a Lab with remote storage and a Ray runner backend:
lab = labtech.Lab(
    # labtech-dev-bucket is the name of our localstack bucket:
    storage=S3fsStorage('labtech-dev-bucket/lab_cache'),
    runner_backend=RayRunnerBackend(),
)

results = lab.run_tasks(experiments, bust_cache=True)
print(results)


# Shutdown the connection to the Ray cluster:
ray.shutdown()
```


## Ray remote function options

<!--
    This section is linked to throughout this file and from an error message
    in RayRunnerBackend.build_runner, so change the name with care.
-->

Ray allows you to configure a number of [options that control how a
task will be executed](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html).
These can be configured for Labtech tasks by defining a
`runner_options()` method on a task type that returns a
`ray.remote_options` section.

For example, you can [configure the minimum memory and CPU cores](https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html) that
must be available to a Ray worker that is executing a task:

```python
    @task
    class Experiment:
        ...

        def runner_options(self):
            # Require 2 CPU cores and 2G of memory for each task of this type.
            return {
                'ray': {
                    'remote_options': {
                        'num_cpus': '2',
                        'memory': (2 * (1024 ** 3)),
                    },
                }
            }

        def run(self):
            ...
```


## Syncing Python environments across a cluster

One of the challenges of running tasks across a distributed cluster is
ensuring that the Python execution environment is identical in each
worker process running on each node. You should employ the following
mechanisms provided by Ray to ensure that your tasks execute
identically wherever they are run.

### Worker initialisation

You can use Ray's [`worker_process_setup_hook`](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html#customizing-worker-process-loggers)
to execute one-off setup code before any tasks are run in a Ray worker process.

For example, you can pass `worker_process_setup_hook` into
`ray.init()` to configure mlflow in each worker:

```python
def worker_setup():
    # Initialise mlflow on each worker to use a centralised
    # mlflow tracking server:
    mlflow.set_tracking_uri('http://my-mlflow-host:8080')
    mlflow.set_experiment('example_ray_experiment')


ray.init(
    runtime_env={
        'worker_process_setup_hook': worker_setup,
    },
)
```

### File and package dependencies

There are two broad approaches to ensuring the necessary source code,
data files, and Python packages are installed on each node in your
Ray cluster, as discussed in Ray's documentation on [Environment
Dependencies](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html):

1. Pre-install all dependencies onto each node in your cluster. Ray's
   cluster launcher [has options to support this](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#preparing-an-environment-using-the-ray-cluster-launcher).
2. Specify a [runtime environment](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments)
   for Ray to install on-demand whenever a node runs a task.
     * When using Labtech for experimentation, your code and
       dependencies may be changing frequently, so the flexibility of a
       runtime environment may be a better fit.
     * You can specify a runtime environment for all tasks with
       `ray.init(runtime_env={...})` or for specific tasks with the
       `runtime_env` [remote function option](#ray-remote-function-options).
     * You can use a runtime environment to:
         * [Share a working directory of local source code and data files](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-local-files)
         * [Share local Python modules that you may still be modifying](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-conda-or-pip-packages)
         * [List Python packages to be installed](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#library-development)


## Fault tolerance

Because a variety of failures can occur in a distributed system, Ray
has [several mechanisms for fault tolerance](https://docs.ray.io/en/latest/ray-core/fault-tolerance.html).

You should ensure that your tasks can be safely executed multiple
times, as Ray may re-execute tasks under certain circumstances. You
can control how many times Ray will re-execute tasks in some of these
circumstances through [remote function options](#ray-remote-function-options):

* If your task's `run()` method raises an exception, Ray will not
  re-execute the task unless you set
  [`retry_exceptions`](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html#retrying-failed-tasks)
  to `True` or a list of exception types.
* If the [worker running a task dies](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html#retrying-failed-tasks)
  or a [stored object is lost due to node failure](https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html)
  then Ray will re-execute the task up to
  [`max_retries`](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html#retrying-failed-tasks)
  (which you can disable by [setting](#ray-remote-function-options) `max_retries=0`).
    * If a task is re-executed after a stored object is lost, it will still
      re-run the task instead of loading it's result from the cache.
* If [Ray's memory monitor](https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html)
  terminates a task to avoid running out of memory, then the task will
  be re-executed irrespective of the `max_retries` setting. You can
  disable this behaviour by disabling Ray's built-in memory monitor.


## Other considerations

* A Lab's `max_workers` must be set to `None` when using the
  [`RayRunnerBackend`][labtech.runners.ray.RayRunnerBackend]. This is
  because Ray concurrency is not limited by a maximum number of tasks
  but by
  [specifying the resource requirements of each task](#ray-remote-function-options).
* Log messages from tasks are not displayed directly under the
  Labtech progress bars but are instead available in Ray's worker
  logs, which are, by default, available from:
    * The standard output stream of your Python script running Labtech
    * Under the cell that executes `ray.init()` in a Jupyter notebook
    * The [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
      that aggregates all worker logs
* Labtech's task monitor is currently unable to report on the CPU
  usage, memory usage, and other process metrics for tasks run on Ray
  clusters.
    * Instead, you can refer to the resource utilisation of cluster
      nodes from the [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
      or enable the [Ray Metrics View](https://docs.ray.io/en/latest/ray-observability/getting-started.html#dash-metrics-view)
      (which requires configuring Prometheus and Grafana).


## API Reference

::: labtech.runners.ray.RayRunnerBackend
    options:
        heading_level: 3
