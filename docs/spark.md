Labtech is a great choice for managing concurrent pipelines of
experiments and other tasks on an [Apache Spark](https://spark.apache.org/) cluster.
There are two primary ways of using Labtech with PySpark:

1. Use [`ThreadRunnerBackend` with PySpark](#threadrunnerbackend-with-pyspark)
   if your tasks primarily involve PySpark-native operations, including
   PySpark data transformations,
   [MLLib](https://spark.apache.org/mllib/), or
   [GraphX](https://spark.apache.org/graphx/).
2. Use [`RayRunnerBackend` on PySpark](#rayrunnerbackend-on-pyspark) if
   your tasks primarily involve native Python code, such as non-PySpark
   data processing in [Pandas](https://pandas.pydata.org/) or
   [Polars](https://pola.rs/) or machine learning with libraries like
   [scikit-learn](https://scikit-learn.org/).


## `ThreadRunnerBackend` with PySpark

If your tasks delegate heavy processing to PySpark-native operations,
then you can use the
[`ThreadRunnerBackend`][labtech.runners.thread.ThreadRunnerBackend] to
run each task on a lightweight thread. Each task's thread can initiate
the PySpark execution for that task and then allow other task threads to
run while waiting for a result to be returned by Spark:

```python
# Assuming we already have a PySpark session available in variable: spark
import labtech
from labtech.runners import ThreadRunnerBackend
from pyspark.sql import functions as sf


@labtech.task
class Experiment:
    table: str
    power: int

    def run(self):
        return (
            spark.read.table(self.table)
            .withColumn('raised_value', sf.col('value') ** self.power)
            .select(sf.mean(sf.col('raised_value')).alias('mean'))
            # We run collect() so that the Spark execution of transformations
            # is triggered from the task's thread. Other task threads will be
            # free to run while this thread waits for Spark to finish executing.
            .collect()[0]['mean']
        )


# Prepare a DataFrame in Spark that can be referenced by name from each task.
table_name = 'dataset'
spark_df = spark.createDataFrame([
    {'value': value} for value in range(1000)
])
spark_df.createOrReplaceTempView(table_name)

experiments = [
    Experiment(
        table=table_name,
        power=power,
    )
    for power in range(10)
]

lab = labtech.Lab(
    runner_backend=ThreadRunnerBackend(),
    # Max workers should be set relative to available Spark cores
    # and the number of cores that each task can leverage.
    max_workers=3,
    storage='storage/spark_lab',
)
results = lab.run_tasks(experiments)
```

### Tips

* Using Spark table names in task parameters and results allows data
  to be efficiently managed in Spark without needed to be serialised,
  transferred, deserialised, and loaded into the main Python process
  when passed between tasks.
    * However, be sure to use table names with random identifiers for
      task outputs to ensure each unique run of a task is output into
      a unique table.
    * You may also like to implement a [custom storage provider](./caching.md#custom-storage) to
      generically handle persistence of Spark tables for task outputs.
* This approach to using Labtech with PySpark is compatible with
  remote [Spark Connect](https://spark.apache.org/spark-connect/)
  connections, as long as your tasks only require [Connect-compatible
  PySpark APIs](https://spark.apache.org/docs/latest/spark-connect-overview.html#what-is-supported),
  otherwise you will need a full Spark Classic connection.


## `RayRunnerBackend` on PySpark

If your tasks perform heavy processing in native Python code that
cannot be readily delegated to Spark, then the recommended approach is
to [start a Ray cluster on PySpark](https://community.databricks.com/t5/technical-blog/ray-on-spark-a-practical-architecture-and-setup-guide/ba-p/127511)
to use with Labtech's
[`RayRunnerBackend`](http://localhost:8000/distributed/).

While Python code can technically be run directly on Spark workers, it
requires expensive serialisation and deserialisation of Python objects
(such as your tasks' parameters and results). [Ray](https://www.ray.io/)
is specifically designed for efficiently distributing Python code
across a cluster of machines, and is [recommended by platforms like
Databricks for distributing native Python tasks](https://docs.databricks.com/aws/en/machine-learning/ray/spark-ray-overview).

The following code provides a basic example of starting a Ray cluster
and using it with Labtech's
[`RayRunnerBackend`][labtech.runners.ray.RayRunnerBackend]. For more
detail, refer to the documentation for [using Labtech with Ray](distributed.md) and
[running Ray on PySpark](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/spark.html).

```python
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

import labtech
from labtech.storage import LocalStorage
from labtech.runners.ray import RayRunnerBackend


# Assuming we already have a running PySpark session,
# we can start a Ray cluster on top of Spark:
setup_ray_cluster(
  max_worker_nodes=2,
  num_cpus_worker_node=2,
  num_gpus_worker_node=0,
  memory_worker_node=(10 * 1024**3),  # 10 GiB
)
ray.init()

...

lab = labtech.Lab(
    storage=LocalStorage(
        'storage/ray_on_spark_lab',
        # We can specify an alternative path if the storage file share
        # is mounted at a different location on the Spark workers:
        runner_dir='/opt/spark/shared-storage/ray_on_spark_lab',
    ),
    runner_backend=RayRunnerBackend(),
)

...

shutdown_ray_cluster()
ray.shutdown()
```
