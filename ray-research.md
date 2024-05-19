# Notes

## https://docs.ray.io/en/latest/ray-core/tasks.html

* Each remote function can specify resource requirements (number of
  CPUs or GPUs, memory, available memory):
  https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#resource-requirements
* A result/object ref passed to a task will be treated like a
  dependency - Ray will get that output to the node it needs to be on:
  https://docs.ray.io/en/latest/ray-core/tasks.html#passing-object-refs-to-ray-tasks
* Use ray.wait() to wait for the first completed future:
  https://docs.ray.io/en/latest/ray-core/tasks.html#passing-object-refs-to-ray-tasks
* Tasks can be cancelled:
  https://docs.ray.io/en/latest/ray-core/tasks.html#cancelling-tasks

## https://docs.ray.io/en/latest/ray-core/tasks/nested-tasks.html

* A task can call another task - potentially a nice paradigm than
  explicitly creating a DAG? Harder for us to implement for labtech,
  as it requires running a task and then "shelving" it while its
  dependencies run.

## https://docs.ray.io/en/latest/ray-core/objects.html

* Use an object to represent the context? Or parts of the context?
* Probably use them to store object results to manage sharing results?
  Maybe make that optional vs just loading from Storage?
* "Remote objects are cached in Ray’s distributed shared-memory object
  store, and there is one object store per node in the cluster. In the
  cluster setting, a remote object can live on one or many nodes,
  independent of who holds the object ref(s)."
* "If the object is a numpy array or a collection of numpy arrays, the
  get call is zero-copy and returns arrays backed by shared object
  store memory. Otherwise, we deserialize the object data into a
  Python object."
* "If the current node’s object store does not contain the object, the
  object is downloaded."
* "Objects are tracked via distributed reference counting, and their
  data is automatically freed once all references to the object are
  deleted."
* "You can also pass objects to tasks via closure-capture."

## https://docs.ray.io/en/latest/ray-core/handling-dependencies.html

* "For production usage or non-changing environments, we recommend
  installing your dependencies into a container image and specifying
  the image using the Cluster Launcher. For dynamic environments (e.g.
  for development and experimentation), we recommend using runtime
  environments."
* "You can build all your files and dependencies into a container
  image and specify this in your your Cluster YAML Configuration." ...
  "You can push local files to the cluster using ray rsync_up"
* Runtime Environments:
  * Can be different for each task, or one for the entire "job"
    (app) - probably just use the latter for simplicity?
  * `ray.init(runtime_env={...})`
  * Specifying a `working_dir` will ensure that local directory is
    pushed to the cluster by `ray.init()`.
  * You can specify pip or conda deps, but probably best to leave
    these to the cluster server setup - otherwise you'll be
    re-downloading every time.
  * You can specify Python modules that your tasks depend on with
    `py_modules`.
    * This can be a directory of python files. "if the local directory
      contains a .gitignore file, the files and paths specified there
      are not uploaded to the cluster."
    * "Note: This feature is currently limited to modules that are
      packages with a single directory containing an __init__.py file.
      For single-file modules, you may use working_dir."
    * Also has an `excludes` option.
  * "If runtime_env cannot be set up (e.g., network issues, download
    failures, etc.), Ray will fail to schedule tasks/actors that
    require the runtime_env. If you call ray.get, it will raise
    RuntimeEnvSetupError with the error message in detail."

## https://docs.ray.io/en/latest/ray-core/scheduling/index.html

* You can choose a different scheduling strategy.

## https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html

* Ray can handle GPU allocation

## https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html

* Ray automatically kills tasks to prevent out-of-memory
* Tasks killed by the memory monitor will be retried infinitely with
  exponential backoff up to 60 seconds.

## https://docs.ray.io/en/latest/ray-core/fault-tolerance.html

* Generally, don't call `.put()` inside a task it seems.

## https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html

* TODO

## https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html

* TODO

## https://docs.ray.io/en/latest/ray-observability/getting-started.html

* Ray Dashboard can be run to view running tasks
* It is automatically run on the head node on a given port.
  * We should make sure tasks are meaningfully named to help our users
    view this.

## https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html

* There is also a CLI to get task status

## https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html

* Log files go into a tmp directory by default
* "By default, Worker stdout and stderr for Tasks and Actors stream to
  the Ray Driver (the entrypoint script that calls ray.init). It helps
  users aggregate the logs for the distributed Ray application in a
  single place."
  * How will we make this work with our own logging? Plug on top of
    this logging and manage it's output ourselves?

## https://docs.ray.io/en/latest/ray-core/ray-dag.html

* There is an API to build a DAG that can be executed, but probably
  not helpful for us.

## https://docs.ray.io/en/latest/ray-core/miscellaneous.html#running-large-ray-clusters

* Tips for running Ray clusters with 1k+ nodes!

## https://docs.ray.io/en/latest/ray-core/user-spawn-processes.html

* Notes for avoiding zombies when your task contains subprocesses and
  is killed.
* "RAY_kill_child_processes_on_worker_exit (default true): Only works
  on Linux. If true, the worker kills all direct child processes on
  exit. This won’t work if the worker crashed. This is NOT recursive,
  in that grandchild processes are not killed by this mechanism."
* "RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper
  (default false): Only works on Linux greater than or equal to 3.4.
  If true, Raylet recursively kills any child processes and grandchild
  processes that were spawned by the worker after the worker exits.
  This works even if the worker crashed. The killing happens within 10
  seconds after the worker death."
* "On non-Linux platforms, user-spawned process is not controlled by
  Ray. The user is responsible for managing the lifetime of the child
  processes. If the parent Ray worker process dies, the child
  processes will continue to run."

## https://docs.ray.io/en/latest/ray-core/patterns/index.html

* TODO

## https://docs.ray.io/en/latest/ray-core/tips-for-first-time.html

* TODO

## https://docs.ray.io/en/latest/ray-core/starting-ray.html

* TODO

## https://docs.ray.io/en/latest/cluster/getting-started.html#cluster-index

* TODO

# Questions

* Can I dynamically create a remote function?
* How will this work with mlflow?
* Have an option to not use distributed storage, where we load and
  save all results locally, and distribute them as shared objects?
