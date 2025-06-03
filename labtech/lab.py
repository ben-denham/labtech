"""Lab and related utilities responsible for running tasks."""
from __future__ import annotations

import concurrent.futures.process
import math
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from tqdm.contrib.logging import logging_redirect_tqdm

from .exceptions import LabError, TaskNotFound
from .monitor import TaskMonitor
from .runners import ForkPerTaskRunnerBackend, ForkPoolRunnerBackend, SerialRunnerBackend, SpawnPoolRunnerBackend, ThreadRunnerBackend
from .storage import LocalStorage, NullStorage
from .tasks import get_direct_dependencies
from .types import ResultMeta, is_task, is_task_type
from .utils import OrderedSet, get_supported_start_methods, is_ipython, logger, tqdm, tqdm_notebook

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any

    from .types import LabContext, ResultT, RunnerBackend, Storage, Task, TaskT
    from .utils import base_tqdm


def check_tasks(tasks: Sequence[Task]) -> None:
    for task in tasks:
        if not is_task(task):
            raise LabError(
                (f'`{repr(task)}` is not a task, please provide a task object '
                 'made from a class decorated with @labtech.task.')
            )


def check_task_types(task_types: Sequence[type[Task]]) -> None:
    for task_type in task_types:
        if not is_task_type(task_type):
            raise LabError(
                (f'`{repr(task_type)}` is not a task type, please provide a '
                 'class decorated with @labtech.task.')
            )


class TaskState:

    def __init__(self, *, coordinator: TaskCoordinator, tasks: Sequence[Task]):
        self.coordinator = coordinator

        self.pending_tasks: OrderedSet[Task] = OrderedSet()
        self.processed_task_ids: set[int] = set()
        self.task_to_direct_dependencies: dict[Task, set[Task]] = defaultdict(set)
        self.task_to_pending_dependencies: dict[Task, set[Task]] = defaultdict(set)
        self.task_to_pending_dependents: dict[Task, set[Task]] = defaultdict(set)
        self.type_to_active_tasks: dict[type[Task], set[Task]] = defaultdict(set)
        # We need to track all unique instances of a task (i.e. that
        # hash the same, but have different identities) so that we can
        # ensure all are updated. Duplicated task instances may occur
        # in dependencies of different tasks.
        self.task_to_instances: dict[Task, list[Task]] = defaultdict(list)

        self.process_tasks(tasks)
        self.check_cyclic_dependences()

    def process_tasks(self, tasks: Iterable[Task]):
        all_dependencies: list[Task] = []
        for task in tasks:
            if id(task) in self.processed_task_ids:
                continue
            self.processed_task_ids.add(id(task))

            dependency_tasks: OrderedSet[Task] = OrderedSet()
            if not self.coordinator.use_cache(task):
                dependency_tasks = OrderedSet(get_direct_dependencies(task, all_identities=False))

            # We insert all of the top-level tasks before processing
            # discovered dependencies, so that we will attempt to run
            # higher-level tasks as soon as possible.
            self.insert_task(task, dependency_tasks)
            all_dependencies += dependency_tasks
        if len(all_dependencies) > 0:
            self.process_tasks(all_dependencies)

    def insert_task(self, task: Task, dependencies: Iterable[Task]):
        self.pending_tasks.add(task)
        self.task_to_instances[task].append(task)
        for dependency in dependencies:
            self.task_to_direct_dependencies[task].add(dependency)
            self.task_to_pending_dependencies[task].add(dependency)
            self.task_to_pending_dependents[dependency].add(task)

    def start_task(self, task: Task):
        self.pending_tasks.remove(task)
        self.type_to_active_tasks[type(task)].add(task)

    def complete_task(self, task: Task, *, result_meta: ResultMeta | None) -> OrderedSet[Task]:
        # A result_meta of None indicates task failure
        if result_meta is not None:
            for task_instance in self.task_to_instances[task]:
                task_instance._set_result_meta(result_meta)

        self.type_to_active_tasks[type(task)].remove(task)
        for dependent in self.task_to_pending_dependents[task]:
            self.task_to_pending_dependencies[dependent].remove(task)

        tasks_with_removable_results: OrderedSet[Task] = OrderedSet()
        for dependency in self.task_to_direct_dependencies[task]:
            self.task_to_pending_dependents[dependency].remove(task)
            if len(self.task_to_pending_dependents[dependency]) == 0:
                tasks_with_removable_results.add(dependency)

        if len(self.task_to_pending_dependents[task]) == 0:
            tasks_with_removable_results.add(task)

        return tasks_with_removable_results

    def get_ready_tasks(self) -> Sequence[Task]:
        ready_tasks = []
        task_type_counts = Counter({
            task_type: len(active_tasks)
            for task_type, active_tasks in self.type_to_active_tasks.items()
        })
        for task in self.pending_tasks:
            if len(self.task_to_pending_dependencies.get(task, set())) > 0:
                continue
            if (task._lt.max_parallel is not None) and (task_type_counts[type(task)] >= task._lt.max_parallel):
                continue
            task_type_counts[type(task)] += 1
            ready_tasks.append(task)
        return ready_tasks

    def check_cyclic_dependences(self):
        visited = {task: False for task in self.pending_tasks}

        def check_cycle(task: Task, parents: set[Task]):
            for dependency in self.task_to_direct_dependencies[task]:
                if dependency == task or dependency in parents:
                    raise LabError(f"Cyclic dependency: Task '{dependency}' depends on itself.")
                elif not visited[dependency]:
                    visited[dependency] = True
                    check_cycle(dependency, (parents | {task}))

        for task in self.pending_tasks:
            if visited[task]:
                continue
            visited[task] = True
            check_cycle(task, set())


class TaskCoordinator:

    def __init__(self, lab: Lab, bust_cache: bool, disable_progress: bool,
                 disable_top: bool, top_format: str, top_sort: str, top_n: int):
        self.lab = lab
        self.bust_cache = bust_cache
        self.disable_progress = disable_progress
        self.disable_top = disable_top
        self.top_format = top_format
        self.top_sort = top_sort
        self.top_n = top_n

    def get_pbar(self, *, task_type: type[Task], task_count: int) -> base_tqdm:
        pbar_func = tqdm_notebook if self.lab.notebook else tqdm
        return pbar_func(
            desc=task_type.__qualname__,
            unit='tasks',
            total=task_count,
            bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}] |{bar}| {percentage:3.0f}% [{remaining} remaining]',
            disable=self.disable_progress,
            dynamic_ncols=True,
        )

    def handle_failure(self, *, ex: Exception, message: str):
        # Simplify subprocess error tracebacks by reporting the cause directly.
        if isinstance(ex.__cause__, concurrent.futures.process._RemoteTraceback):
            ex = ex.__cause__

        if self.lab.continue_on_failure:
            logger.error(f"{message} Skipping task. Failure cause: {ex}")
        else:
            logger.error(message)
            lab_error = LabError(str(ex))
            raise lab_error from ex

    def use_cache(self, task: Task) -> bool:
        return (not self.bust_cache) and self.lab.is_cached(task)

    def run(self, tasks: Sequence[Task]) -> dict[Task, Any]:
        task_results = {}

        state = TaskState(coordinator=self, tasks=tasks)
        task_type_counts = Counter([type(task) for task in state.pending_tasks])
        task_type_max_digits = {
            task_type: math.ceil(math.log10(task_type_counts[task_type]))
            for task_type, task_type_count
            in task_type_counts.items()
        }
        task_type_to_task_count = {task_type: 0 for task_type in task_type_counts.keys()}
        pbars = {
            task_type: self.get_pbar(
                task_type=task_type,
                task_count=task_count,
            )
            for task_type, task_count in task_type_counts.items()
        }

        runner = self.lab.runner_backend.build_runner(
            context=self.lab.context,
            max_workers=self.lab.max_workers,
            storage=self.lab._storage,
        )

        task_monitor = None
        if not self.disable_top:
            task_monitor = TaskMonitor(
                runner=runner,
                top_format=self.top_format,
                top_sort=self.top_sort,
                top_n=self.top_n,
                notebook=self.lab.notebook,
            )
            task_monitor.show()

        last_monitor_update = monotonic()

        def process_completed_tasks():
            nonlocal last_monitor_update

            # Wait up to a short delay before allowing the
            # task monitor to update.
            for task, res in runner.wait(timeout_seconds=0.5):
                if isinstance(res, Exception):
                    tasks_with_removable_results = state.complete_task(task, result_meta=None)
                    self.handle_failure(ex=res, message=f"Task '{task}' failed.")
                elif isinstance(res, ResultMeta):
                    if task in tasks:
                        task_results[task] = runner.get_result(task).value
                    tasks_with_removable_results = state.complete_task(task, result_meta=res)
                    pbars[type(task)].update(1)
                    pbars[type(task)].refresh(nolock=True)
                else:
                    raise LabError(f'Unexpected task res type: {type(res)}')

                runner.remove_results(tasks_with_removable_results)

            # Update task monitor at most every half second.
            if task_monitor is not None and ((monotonic() - last_monitor_update) >= 0.5):
                task_monitor.update()
                last_monitor_update = monotonic()

        redirected_loggers = [] if self.lab.notebook else [logger]
        with logging_redirect_tqdm(loggers=redirected_loggers):
            try:
                try:
                    while (len(state.pending_tasks) > 0) or (runner.pending_task_count() > 0):
                        ready_tasks = state.get_ready_tasks()
                        for task in ready_tasks:
                            state.start_task(task)
                            task_type_to_task_count[type(task)] += 1
                            task_number = (str(task_type_to_task_count[type(task)])
                                           .zfill(task_type_max_digits[type(task)]))
                            runner.submit_task(
                                task,
                                task_name=f'{type(task).__name__}[{task_number}]',
                                use_cache=self.use_cache(task),
                            )
                        process_completed_tasks()
                except KeyboardInterrupt as first_keyboard_interrupt:
                    logger.info(('Interrupted. Finishing running tasks. '
                                 'Press Ctrl-C again to terminate running tasks immediately (may disrupt result caching or other important actions).'))
                    try:
                        runner.cancel()
                        # Process completed tasks until running tasks
                        # have completed.
                        while runner.pending_task_count() > 0:
                            process_completed_tasks()
                    except KeyboardInterrupt:
                        logger.info('Terminating running tasks.')
                        runner.stop()
                        # Process completed tasks one last time after
                        # tasks have been killed.
                        process_completed_tasks()
                        raise
                    else:
                        raise first_keyboard_interrupt
                else:
                    return task_results
            finally:
                if task_monitor is not None:
                    task_monitor.update()
                runner.close()
                for pbar in pbars.values():
                    pbar.close()
                if task_monitor is not None:
                    task_monitor.close()


class DefaultStorage(StrEnum):
    singleton = 'labtech_storage'


default_storage = DefaultStorage.singleton


class Lab:
    """Primary interface for configuring, running, and getting results of tasks.

    A Lab can be used to run tasks with [`run_tasks()`][labtech.Lab.run_tasks].

    Previously cached tasks can be retrieved with
    [`cached_tasks()`][labtech.Lab.cached_tasks], and can then have
    their results loaded with [`run_tasks()`][labtech.Lab.run_tasks]
    or be removed from the cache storage with
    [`uncache_tasks()`][labtech.Lab.uncache_tasks].

    """

    def __init__(self, *,
                 storage: str | Path | None | Storage | DefaultStorage = default_storage,
                 continue_on_failure: bool = True,
                 max_workers: int | None = None,
                 notebook: bool | None = None,
                 context: LabContext | None = None,
                 runner_backend: str | RunnerBackend | None = None):
        """
        Args:
            storage: Where task results should be cached to. A string or
                [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
                will be interpreted as the path to a local directory, `None`
                will result in no caching. Any [Storage][labtech.types.Storage]
                instance may also be specified. Defaults to a "labtech_storage"
                directory inside the current working directory.
            continue_on_failure: If `True`, exceptions raised by tasks will be
                logged, but execution of other tasks will continue.
            max_workers: The maximum number of parallel worker processes for
                running tasks. A sensible default will be determined by the
                runner_backend (`'fork'` and `'spawn'` use the number of
                processors on the machine given by `os.cpu_count()`, while
                `'thread'` uses `os.cpu_count() + 4`).
            notebook: Determines whether to use notebook-friendly graphical
                progress bars. When set to `None` (the default), labtech will
                detect whether the code is being run from an IPython notebook.
            context: A dictionary of additional variables to make available to
                tasks. The context will not be cached, so the values should not
                affect results (e.g. parallelism factors) or should be kept
                constant between runs (e.g. datasets).
            runner_backend: Controls how tasks are run in parallel. It can
                optionally be set to one of the following options:

                * `'spawn'`: Uses the
                  [`SpawnPoolRunnerBackend`][labtech.runners.SpawnPoolRunnerBackend]
                  to run tasks on a pool of spawned subprocesses. The
                  context and dependency task results are
                  copied/duplicated into the memory of each
                  subprocess. The default on macOS and Windows when
                  `max_workers > 1`.
                * `'fork'`: Uses the
                  [`ForkPoolRunnerBackend`][labtech.runners.ForkPoolRunnerBackend]
                  to run tasks on a pool of forked subprocesses.
                  Memory use is reduced by sharing the context between
                  tasks with memory inherited from the parent process.
                  The default on platforms that support forked Python
                  subprocesses when `max_workers > 1`: Linux and other
                  POSIX systems, but not macOS or Windows.
                * `'fork-per-task'`: Uses the
                  [`ForkPerTaskRunnerBackend`][labtech.runners.ForkPerTaskRunnerBackend]
                  to run each task in a forked subprocess. Shares
                  dependency task results as well as the context in
                  memory shared between subprocesses but at the cost
                  of forking a new subprocess for each task. Best used
                  when dependency task results are large compared to
                  the overall number of tasks.
                * `'thread'`: Uses the
                  [`ThreadRunnerBackend`][labtech.runners.ThreadRunnerBackend]
                  to run each task in a separate Python thread. Because
                  [Python threads do not execute in parallel](https://docs.python.org/3/glossary.html#term-global-interpreter-lock),
                  this runner is best suited for running tasks that are
                  constrained by non-blocking IO operations (e.g. web
                  requests), or for running a single worker with live task
                  monitoring. Memory use is reduced by sharing the same
                  in-memory context and dependency task results across
                  threads. The default when `max_workers = 1`.
                * `'serial'`: Uses the
                  [`SerialRunnerBackend`][labtech.runners.SerialRunnerBackend]
                  to run each task serially in the main process and thread.
                  The task monitor will only be updated in between tasks. Mainly
                  useful when troubleshooting issues running tasks on different
                  threads and processes.
                * Any instance of a
                  [`RunnerBackend`][labtech.types.RunnerBackend],
                  allowing for custom task management implementations.

                For details on the differences between `'fork'` and
                `'spawn'` backends, see [the Python documentation on
                `multiprocessing` start methods](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods).

        """
        if isinstance(storage, DefaultStorage):
            logger.warning(f'Caching labtech results in a local "{storage.value}" directory. Construct Lab() with a storage argument to suppress this warning.')
            storage = LocalStorage(storage.value)
        elif isinstance(storage, str) or isinstance(storage, Path):
            storage = LocalStorage(storage)
        elif storage is None:
            storage = NullStorage()
        self._storage = storage
        self.continue_on_failure = continue_on_failure
        self.max_workers = max_workers
        self.notebook = is_ipython() if notebook is None else notebook
        if context is None:
            context = {}
        self.context = context
        if runner_backend is None:
            start_methods = get_supported_start_methods()
            if self.max_workers == 1:
                runner_backend = ThreadRunnerBackend()
            elif 'fork' in start_methods:
                runner_backend = ForkPoolRunnerBackend()
            elif 'spawn' in start_methods:
                runner_backend = SpawnPoolRunnerBackend()
            else:
                raise LabError(('Default \'fork\' and \'spawn\' multiprocessing runner '
                                'backends are not supported on your system.'
                                'Please specify a system-compatible runner_backend.'))
        elif isinstance(runner_backend, str):
            if runner_backend == 'fork':
                runner_backend = ForkPoolRunnerBackend()
            elif runner_backend == 'fork-per-task':
                runner_backend = ForkPerTaskRunnerBackend()
            elif runner_backend == 'spawn':
                runner_backend = SpawnPoolRunnerBackend()
            elif runner_backend == 'serial':
                runner_backend = SerialRunnerBackend()
            elif runner_backend == 'thread':
                runner_backend = ThreadRunnerBackend()
            else:
                raise LabError(f'Unrecognised runner_backend: {runner_backend}')
        self.runner_backend = runner_backend

    def run_tasks(self, tasks: Sequence[TaskT], *,
                  bust_cache: bool = False,
                  disable_progress: bool = False,
                  disable_top: bool = False,
                  top_format: str = '$name $status since $start_time CPU: $cpu MEM: $rss',
                  top_sort: str = 'start_time',
                  top_n: int = 10) -> dict[TaskT, Any]:
        """Run the given tasks with as much process parallelism as possible.
        Loads task results from the cache storage where possible and
        caches results of executed tasks.

        Any attribute of a task that is itself a task object is
        considered a "nested task", and will be executed or loaded so
        that it's result is made available to its parent task. If the
        same task is nested inside multiple task objects, it will only
        be executed/loaded once.

        As well as returning the results, each task's result will be
        assigned to a `result` attribute on the task itself.

        Args:
            tasks: The tasks to execute. Each should be an instance of a class
                decorated with [`labtech.task`][labtech.task].
            bust_cache: If `True`, no task results will be loaded from the
                cache storage; all tasks will be re-executed.
            disable_progress: If `True`, do not display a tqdm progress bar
                tracking task execution.
            disable_top: If `True`, do not display the list of top active tasks.
            top_format: Format for each top active task. Follows the format
                rules for
                [template strings](https://docs.python.org/3/library/string.html#template-strings)
                and may include any of the following attributes for
                substitution:

                * `name`: The task's name displayed in logs.
                * `pid`: The task's primary process id.
                * `status`: Whether the task is being run or loaded from cache.
                * `start_time`: The time the task's primary process started.
                * `children`: The number of child tasks of the task's primary
                  process.
                * `threads`: The number of active CPU threads for the task
                  (including across any child processes).
                * `cpu`: The CPU percentage (1 core = 100%) being used by the
                  task (including across any child processes).
                * `rss`: The resident set size (RSS) memory percentage being
                  used by the task (including across any child processes). RSS
                  is the primary measure of memory usage.
                * `vms`: The virtual memory size (VMS) percentage being used by
                  the task (including across any child processes).
            top_sort: Sort order for the top active tasks. Can be any of the
                attributes available for use in `top_format`. If the string is
                preceded by a `-`, the sort order will be reversed. Defaults to
                showing oldest tasks first.
            top_n: The maximum number of top active tasks to display.

        Returns:
            A dictionary mapping each of the provided tasks to its
                corresponding result.

        """
        check_tasks(tasks)

        for task in tasks:
            if task.code_version != task.current_code_version:
                raise LabError(
                    (f'`{repr(task)}` cannot be run, as it has code_version={task.code_version!r} '
                     f'while the current implementation of {task.__class__.__name__} has '
                     f'code_version={task.current_code_version!r}. You should construct new '
                     f'{task.__class__.__name__} tasks to run instead of running tasks loaded from cache.')
                )


        coordinator = TaskCoordinator(
            self,
            bust_cache=bust_cache,
            disable_progress=disable_progress,
            disable_top=disable_top,
            top_format=top_format,
            top_sort=top_sort,
            top_n=top_n,
        )
        results = coordinator.run(tasks)

        failed_tasks = {task for task in tasks if task not in results}
        if failed_tasks:
            raise LabError(f'Failed to complete {len(failed_tasks)} submitted task(s)')

        # Return results in the same order as tasks
        return {task: results[task] for task in tasks}

    def run_task(self, task: Task[ResultT], **kwargs) -> ResultT:
        """Run a single task and return its result. Supports the same keyword
        arguments as `run_tasks`.

        NOTE: If you have many tasks to run, you should use
        `run_tasks` instead to parallelise their execution.

        Returns:
            The result of the given task.

        """
        results = self.run_tasks([task], **kwargs)
        return results[task]

    def cached_tasks(self, task_types: Sequence[type[TaskT]]) -> Sequence[TaskT]:
        """Returns all task instances present in the Lab's cache storage for
        the given `task_types`, each of which should be a task class
        decorated with [`labtech.task`][labtech.task].

        Does not load task results from the cache storage, but they
        can be loaded by calling
        [`run_tasks()`][labtech.Lab.run_tasks] with the returned task
        instances.

        """
        check_task_types(task_types)
        keys = self._storage.find_keys()
        tasks = []
        for key in keys:
            for task_type in task_types:
                try:
                    task = task_type._lt.cache.load_task(self._storage, task_type, key)
                except TaskNotFound:
                    pass
                else:
                    tasks.append(task)
                    break
        return tasks

    def is_cached(self, task: Task) -> bool:
        """Checks if a result is present for given task in the Lab's cache
        storage."""
        check_tasks([task])
        return task._lt.cache.is_cached(self._storage, task)

    def uncache_tasks(self, tasks: Sequence[Task]):
        """Removes cached results for the given tasks from the Lab's cache
        storage."""
        check_tasks(tasks)
        for task in tasks:
            if self.is_cached(task):
                task._lt.cache.delete(self._storage, task)
