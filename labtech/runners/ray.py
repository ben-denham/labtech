from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from labtech.exceptions import RunnerError
from labtech.tasks import get_direct_dependencies
from labtech.types import LabContext, ResultMeta, ResultT, Runner, RunnerBackend, Storage, Task, TaskMonitorInfo, TaskResult
from labtech.utils import logger

from .base import run_or_load_task

try:
    import ray
except ImportError:
    raise ImportError("Failed to import the `ray` library, please run `pip install ray` to enable Labtech\'s Ray support.")


# TODO: How to do fault tolerance?
# * Careful with OOM restarts: https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
# TODO: Tests
# TODO: Example + cookbook

# Ignore type-checking because Ray's typing doesn't handle keyword
# arguments, even though they work.
@ray.remote(num_returns=2)  # type: ignore[arg-type]
def _ray_func(*, task: Task[ResultT], task_name: str, use_cache: bool,
              context: LabContext, storage: Storage, result_refs_map: dict[Task, ray.ObjectRef]) -> tuple[ResultMeta, ResultT]:
    # TODO: Handle logging (including task_name)

    # Get all dependency results from ray object store for local use.
    for dependency_task in get_direct_dependencies(task):
        dependency_task._set_results_map({dependency_task: ray.get(result_refs_map[dependency_task])})

    task_result = run_or_load_task(
        task=task,
        use_cache=use_cache,
        filtered_context=task.filter_context(context),
        storage=storage,
    )
    return (task_result.meta, task_result.value)


@dataclass(frozen=True)
class TaskRefs:
    task: Task
    result_meta_ref: ray.ObjectRef
    result_value_ref: ray.ObjectRef


class RayRunner(Runner):

    def __init__(self, *, context: LabContext, storage: Storage) -> None:
        if not ray.is_initialized():
            raise RunnerError('Ray has not yet been initialized. Please call `ray.init()` to start or connect to a Ray cluster.')

        logger.debug('Uploading context and storage objects to ray object store')
        self.context_ref = ray.put(context)
        self.storage_ref = ray.put(storage)
        logger.debug('Uploaded context and storage objects to ray object store')

        self.cancelled = False
        self.pending_refs_lookup: dict[ray.ObjectRef, TaskRefs] = {}
        self.result_refs_map: dict[Task, TaskRefs] = {}

    def submit_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        options = task.runner_options().get('ray', {}).get('remote_options', {})
        result_refs: tuple[ray.ObjectRef, ray.ObjectRef] = (
            # Ignore incorrect handling of multiple returns in Ray's typing.
            _ray_func  # type: ignore[assignment]
            .options(**options)
            .remote(
                # Ignore type-checking because Ray's typing doesn't
                # handle keyword arguments, even though they work.
                task=task,  # type: ignore[call-arg]
                task_name=task_name,  # type: ignore[call-arg]
                use_cache=use_cache,  # type: ignore[call-arg]
                context=self.context_ref,  # type: ignore[call-arg]
                storage=self.storage_ref,  # type: ignore[call-arg]
                result_refs_map=self.result_refs_map,
            )
        )
        result_meta_ref, result_value_ref = result_refs
        self.pending_refs_lookup[result_meta_ref] = TaskRefs(
            task=task,
            result_meta_ref=result_meta_ref,
            result_value_ref=result_value_ref,
        )

    def wait(self, *, timeout_seconds: Optional[float]) -> Iterator[tuple[Task, ResultMeta | BaseException]]:
        result_meta_refs = list(self.pending_refs_lookup.keys())
        done_result_meta_refs, _ = ray.wait(
            result_meta_refs,
            num_returns=1,
            timeout=timeout_seconds,
            fetch_local=True,
        )
        for result_meta_ref in done_result_meta_refs:
            if self.cancelled:
                continue
            task_refs = self.pending_refs_lookup[result_meta_ref]
            task = task_refs.task
            try:
                result_meta = ray.get(result_meta_ref)
            except BaseException as ex:
                yield (task, ex)
            else:
                self.result_refs_map[task] = task_refs
                yield (task, result_meta)
        self.pending_refs_lookup = {
            result_meta_ref: self.pending_refs_lookup[result_meta_ref]
            for result_meta_ref in self.pending_refs_lookup.keys()
            if result_meta_ref not in done_result_meta_refs
        }

    def cancel(self) -> None:
        self.cancelled = True

    def stop(self) -> None:
        for task_refs in self.pending_refs_lookup.values():
            ray.cancel(task_refs.result_meta_ref, force=True)
            ray.cancel(task_refs.result_value_ref, force=True)

    def close(self) -> None:
        pass

    def pending_task_count(self) -> int:
        return len(self.pending_refs_lookup)

    def get_result(self, task: Task) -> TaskResult:
        task_refs = self.result_refs_map[task]
        value, meta = ray.get([
            task_refs.result_value_ref,
            task_refs.result_meta_ref,
        ])
        return TaskResult(value=value, meta=meta)

    def remove_results(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            if task in self.result_refs_map:
                # Removing references to an object allows it to be
                # removed by Ray
                del self.result_refs_map[task]

    def get_task_infos(self) -> list[TaskMonitorInfo]:
        # TODO: Handle task monitoring
        return []


class RayRunnerBackend(RunnerBackend):
    """Runner Backend that runs each task as a task on a [Ray](https://www.ray.io/) cluster.

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

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> Runner:
        if max_workers is not None:
            raise RunnerError((
                'Remove max_workers from your Lab configuration, as RayRunnerBackend only supports max_workers=None. '
                'You can manage Ray concurrency by specifying : TODO_LINK_TO_COOKBOOK.'
            ))

        return RayRunner(
            context=context,
            storage=storage,
        )
