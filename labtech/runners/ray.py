from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from time import monotonic, sleep
from typing import Iterator, Optional, Sequence

from labtech.exceptions import RunnerError
from labtech.tasks import get_direct_dependencies
from labtech.types import LabContext, ResultMeta, ResultT, Runner, RunnerBackend, Storage, Task, TaskMonitorInfo, TaskResult
from labtech.utils import logger

from .base import run_or_load_task

try:
    import ray
    import ray.util.state as ray_state
except ImportError:
    raise ImportError("Failed to import the `ray` library, please run `pip install ray` to enable Labtech\'s Ray support.")

# TODO: Handle logging (including task_name)
# * https://docs.ray.io/en/latest/ray-core/api/doc/ray.LoggingConfig.html#ray.LoggingConfig
# * https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html
# TODO: How to do fault tolerance?
# * Careful with OOM restarts: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
# TODO: Handling Python dependencies across cluster
# TODO: Tests
# TODO: Distributed docs page
# * Mention ray's special handling of numpy arrays
# * For CPU/Memory, point users to the Metrics view on the dashboard, which requires Prometheus and Grafana: https://docs.ray.io/en/latest/ray-observability/getting-started.html#dash-metrics-view
# TODO: Example

@dataclass(frozen=True)
class TaskDetail:
    task: Task
    task_name: str
    result_meta_ref: ray.ObjectRef
    result_value_ref: ray.ObjectRef


# Ignore type-checking because Ray's typing doesn't handle keyword
# arguments, even though they work.
@ray.remote(num_returns=2)  # type: ignore[arg-type]
def _ray_func(*, task: Task[ResultT], task_name: str, use_cache: bool,
              context: LabContext, storage: Storage, result_detail_map: dict[Task, ray.ObjectRef]) -> tuple[ResultMeta, ResultT]:
    # Get all dependency results from ray object store for local use.
    for dependency_task in get_direct_dependencies(task):
        dependency_task._set_results_map({dependency_task: ray.get(result_detail_map[dependency_task])})

    task_result = run_or_load_task(
        task=task,
        use_cache=use_cache,
        filtered_context=task.filter_context(context),
        storage=storage,
    )
    return (task_result.meta, task_result.value)


class RayRunner(Runner):

    def __init__(self, *, context: LabContext, storage: Storage,
                 monitor_interval_seconds: float, monitor_timeout_seconds: int) -> None:
        self.monitor_interval_seconds = monitor_interval_seconds
        self.monitor_timeout_seconds = monitor_timeout_seconds

        if not ray.is_initialized():
            raise RunnerError('Ray has not yet been initialized. Please call `ray.init()` to start or connect to a Ray cluster.')

        logger.debug('Uploading context and storage objects to ray object store')
        self.context_ref = ray.put(context)
        self.storage_ref = ray.put(storage)
        logger.debug('Uploaded context and storage objects to ray object store')

        self.cancelled = False
        self.pending_task_name_to_use_cache: dict[str, bool] = {}
        self.pending_detail_map: dict[ray.ObjectRef, TaskDetail] = {}
        self.result_detail_map: dict[Task, TaskDetail] = {}

        self.current_task_infos: list[TaskMonitorInfo] = []
        self.monitor_thread_running = True
        self.monitor_thread = Thread(target=self._monitor_thread)
        self.monitor_thread.start()

    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        options = task.runner_options().get('ray', {}).get('remote_options', {})
        result_refs: tuple[ray.ObjectRef, ray.ObjectRef] = (
            # Ignore incorrect handling of multiple returns in Ray's typing.
            _ray_func  # type: ignore[assignment]
            .options(**options, name=task_name)
            .remote(
                # Ignore type-checking because Ray's typing doesn't
                # handle keyword arguments, even though they work.
                task=task,  # type: ignore[call-arg]
                task_name=task_name,  # type: ignore[call-arg]
                use_cache=use_cache,  # type: ignore[call-arg]
                context=self.context_ref,  # type: ignore[call-arg]
                storage=self.storage_ref,  # type: ignore[call-arg]
                result_detail_map=self.result_detail_map,
            )
        )
        result_meta_ref, result_value_ref = result_refs
        self.pending_task_name_to_use_cache[task_name] = use_cache
        self.pending_detail_map[result_meta_ref] = TaskDetail(
            task=task,
            task_name=task_name,
            result_meta_ref=result_meta_ref,
            result_value_ref=result_value_ref,
        )

    def wait(self, *, timeout_seconds: Optional[float]) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        result_meta_refs = list(self.pending_detail_map.keys())
        done_result_meta_refs, _ = ray.wait(
            result_meta_refs,
            num_returns=1,
            timeout=timeout_seconds,
            fetch_local=True,
        )
        for result_meta_ref in done_result_meta_refs:
            task_detail = self.pending_detail_map[result_meta_ref]
            del self.pending_task_name_to_use_cache[task_detail.task_name]

            if self.cancelled:
                continue

            task = task_detail.task
            try:
                result_meta = ray.get(result_meta_ref)
            except BaseException as ex:
                yield (task, ex)
            else:
                self.result_detail_map[task] = task_detail
                yield (task, result_meta)
        self.pending_detail_map = {
            result_meta_ref: self.pending_detail_map[result_meta_ref]
            for result_meta_ref in self.pending_detail_map.keys()
            if result_meta_ref not in done_result_meta_refs
        }

    def cancel(self) -> None:
        self.cancelled = True

    def stop(self) -> None:
        for task_detail in self.pending_detail_map.values():
            ray.cancel(task_detail.result_meta_ref, force=True)
            ray.cancel(task_detail.result_value_ref, force=True)

    def close(self) -> None:
        self.monitor_thread_running = False
        self.monitor_thread.join()

    def pending_task_count(self) -> int:
        return len(self.pending_detail_map)

    def get_result(self, task: Task) -> TaskResult:
        task_detail = self.result_detail_map[task]
        value, meta = ray.get([
            task_detail.result_value_ref,
            task_detail.result_meta_ref,
        ])
        return TaskResult(value=value, meta=meta)

    def remove_results(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            if task in self.result_detail_map:
                # Removing references to an object allows it to be
                # removed by Ray
                del self.result_detail_map[task]


    def _fetch_task_infos(self) -> list[TaskMonitorInfo]:
        try:
            task_states = [
                task_state
                for running_state in ['RUNNING', 'RUNNING_IN_RAY_GET', 'RUNNING_IN_RAY_WAIT']
                for task_state in ray_state.list_tasks(
                    filters=[('state', '=', running_state)],
                    detail=True,
                    timeout=self.monitor_timeout_seconds,
                )
            ]
        except Exception as ex:
            logger.warning(f'Task monitor failed to query Ray task states: {ex}')
            return []

        task_name_to_task_info = {}
        for task_state in task_states:
            # Avoid duplicating a task if it changed state and was
            # returned in multiple ray state queries.
            if task_state.name in task_name_to_task_info:
                continue

            status = 'unknown'
            use_cache = self.pending_task_name_to_use_cache.get(task_state.name)
            if use_cache is not None:
                status = 'loading' if use_cache else 'running'

            start_time = (datetime.max, 'n/a')
            if task_state.start_time_ms is not None:
                start_datetime = datetime.fromtimestamp(task_state.start_time_ms // 1000)
                start_time = (start_datetime, start_datetime.strftime('%H:%M:%S'))

            task_name_to_task_info[task_state.name] = {
                'name': task_state.name,
                'status': status,
                'start_time': start_time,
                # Getting Process/CPU/Memory stats for Ray tasks is
                # not currently possible:
                # https://discuss.ray.io/t/how-to-programatically-do-real-time-monitoring-of-actor-task-resource-usage-heap-memory-obj-store-memory-cpu/8454
                'pid': 'n/a',
                'children': 'n/a',
                'threads': 'n/a',
                'cpu': 'n/a',
                'rss': 'n/a',
                'vms': 'n/a',
            }

        return list(task_name_to_task_info.values())

    def _monitor_thread(self) -> None:
        next_time = monotonic()
        while self.monitor_thread_running:
            sleep(max(0, next_time - monotonic()))
            self.current_task_infos = self._fetch_task_infos()
            next_time = next_time + self.monitor_interval_seconds

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        return self.current_task_infos


class RayRunnerBackend(RunnerBackend):
    """Runner Backend that runs each task as a task on a [Ray](https://www.ray.io/) cluster.

    Ray's [shared-memory object store](https://docs.ray.io/en/latest/ray-core/objects.html)
    is used to distribute context and results between nodes, and Ray
    will allocate task's to cluster nodes where large memory
    dependencies are already loaded.

    [Ray remote options](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html)
    may be provided for a task by defining a `runner_options()` on
    it's Task type that returns a dictionary of options under
    `ray.remote_options`:

    ```
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

    """

    def __init__(self, monitor_interval_seconds: float = 1, monitor_timeout_seconds: int = 5) -> None:
        """
        Args:
            monitor_interval_seconds: Determines frequency of requests to
                Ray for task states.
            monitor_timeout_seconds: Maximum time to wait for a request to
                Ray for task states.
        """
        self.monitor_interval_seconds = monitor_interval_seconds
        self.monitor_timeout_seconds = monitor_timeout_seconds

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> Runner:
        if max_workers is not None:
            raise RunnerError((
                'Remove max_workers from your Lab configuration, as RayRunnerBackend only supports max_workers=None. '
                'You can manage Ray concurrency by specifying : TODO_LINK_TO_COOKBOOK.'
            ))

        return RayRunner(
            context=context,
            storage=storage,
            monitor_interval_seconds=self.monitor_interval_seconds,
            monitor_timeout_seconds=self.monitor_timeout_seconds,
        )
