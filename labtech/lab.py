"""Lab and related utilities responsible for running tasks."""

from collections import Counter, defaultdict
import concurrent.futures
import concurrent.futures.process
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime
from enum import Enum
import logging
from logging.handlers import QueueHandler
import math
import multiprocessing
from pathlib import Path
import signal
import sys
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Type, Union

from frozendict import frozendict
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.contrib.logging import logging_redirect_tqdm

from .types import Task, TaskT, ResultT, ResultMeta, ResultsMap, TaskResult, Storage, is_task
from .tasks import find_tasks_in_param
from .exceptions import LabError, TaskNotFound
from .utils import OrderedSet, LoggerFileProxy, logger
from .storage import NullStorage, LocalStorage
from .executors import SerialExecutor, wait_for_first_future

_IN_TASK_SUBPROCESS = False


@contextmanager
def optional_mlflow(task: Task):

    def log_params(value: Any, *, path: str = ''):
        prefix = path if path == '' else f'{path}.'
        if is_task(value):
            for field in fields(value):
                log_params(getattr(value, field.name), path=f'{prefix}{field.name}')
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                log_params(item, path=f'{prefix}{i}')
        elif isinstance(value, frozendict):
            for key, item in value.items():
                log_params(item, path=f'{prefix}{key}')
        elif isinstance(value, Enum):
            mlflow.log_param(path, f'{type(value).__qualname__}.{value.name}')
        elif ((value is None)
              or isinstance(value, str)
              or isinstance(value, bool)
              or isinstance(value, float)
              or isinstance(value, int)):
            mlflow.log_param(path, value)
        else:
            raise LabError(
                (f"Unable to mlflow log parameter '{path}' of type '{type(value).__qualname__}' "
                 f"in task of type '{type(task).__qualname__}'.")
            )

    if task._lt.mlflow_run:
        try:
            import mlflow
        except ImportError:
            raise LabError(
                (f"Task type '{type(task).__qualname__}' is configured with mlflow_run=True, but "
                 "mlflow cannot be imported. You can install mlflow with `pip install mlflow`.")
            )
        with mlflow.start_run():
            mlflow.set_tag('labtech_task_type', type(task).__qualname__)
            log_params(task)
            yield
    else:
        yield


class TaskState:

    def __init__(self, *, runner: 'TaskRunner', tasks: Sequence[Task]):
        self.runner = runner
        self.results_map: ResultsMap = {}

        self.pending_tasks: OrderedSet[Task] = OrderedSet()
        self.processed_task_ids: Set[int] = set()
        self.task_to_direct_dependencies: Dict[Task, Set[Task]] = defaultdict(set)
        self.task_to_pending_dependencies: Dict[Task, Set[Task]] = defaultdict(set)
        self.task_to_pending_dependents: Dict[Task, Set[Task]] = defaultdict(set)
        self.type_to_active_tasks: Dict[Type[Task], Set[Task]] = defaultdict(set)
        # We need to track all unique instances of a task (i.e. that
        # hash the same, but have different identities) so that we can
        # ensure all are updated. Duplicated task instances may occur
        # in dependencies of different tasks.
        self.task_to_instances: Dict[Task, List[Task]] = defaultdict(list)

        self.process_tasks(tasks)
        self.check_cyclic_dependences()

    def process_tasks(self, tasks: Iterable[Task]):
        all_dependencies: List[Task] = []
        for task in tasks:
            if id(task) in self.processed_task_ids:
                continue
            self.processed_task_ids.add(id(task))

            dependent_tasks: List[Task] = []
            if not self.runner.use_cache(task):
                # Find all dependent tasks inside task fields
                for field in fields(task):
                    field_value = getattr(task, field.name)
                    dependent_tasks += find_tasks_in_param(field_value)

            # We insert all of the top-level tasks before processing
            # discovered dependencies, so that we will attempt to run
            # higher-level tasks as soon as possible.
            self.insert_task(task, dependent_tasks)
            all_dependencies += dependent_tasks
        if len(all_dependencies) > 0:
            self.process_tasks(all_dependencies)

    def insert_task(self, task: Task, dependencies: Iterable[Task]):
        task._set_results_map(self.results_map)
        self.pending_tasks.add(task)
        self.task_to_instances[task].append(task)
        for dependency in dependencies:
            self.task_to_direct_dependencies[task].add(dependency)
            self.task_to_pending_dependencies[task].add(dependency)
            self.task_to_pending_dependents[dependency].add(task)

    def start_task(self, task: Task):
        self.pending_tasks.remove(task)
        self.type_to_active_tasks[type(task)].add(task)

    def complete_task(self, task: Task, *, task_result: Optional[TaskResult]):
        # A task_result of None indicates task failure
        if task_result is not None:
            for task_instance in self.task_to_instances[task]:
                task_instance._set_result_meta(task_result.meta)
            self.results_map[task] = task_result.value

        self.type_to_active_tasks[type(task)].remove(task)
        for dependent in self.task_to_pending_dependents[task]:
            self.task_to_pending_dependencies[dependent].remove(task)

        for dependency in self.task_to_direct_dependencies[task]:
            self.task_to_pending_dependents[dependency].remove(task)
            if len(self.task_to_pending_dependents[dependency]) == 0:
                self.remove_result(dependency)

        if len(self.task_to_pending_dependents[task]) == 0:
            self.remove_result(task)

    def remove_result(self, task: Task):
        if self.runner.keep_nested_results or task not in self.results_map:
            return
        logger.debug(f"Removing result from in-memory cache for task: '{task}'")
        del self.results_map[task]

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

        def check_cycle(task: Task, parents: Set[Task]):
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


def init_task_worker(log_queue):
    global _IN_TASK_SUBPROCESS
    _IN_TASK_SUBPROCESS = True
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Sub-processes should log onto the queue in order to printed
    # in serial by the main process.
    logger.handlers = []
    logger.addHandler(QueueHandler(log_queue))
    sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')
    sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')


class TaskRunner:

    def __init__(self, lab: 'Lab', bust_cache: bool, keep_nested_results: bool,
                 disable_progress: bool):
        self.lab = lab
        self.bust_cache = bust_cache
        self.keep_nested_results = keep_nested_results
        self.disable_progress = disable_progress
        self.log_queue = multiprocessing.Manager().Queue(-1)

    def get_executor(self, serial: bool = False):
        if (self.lab.max_workers is not None and self.lab.max_workers == 1):
            serial = True

        if serial:
            return SerialExecutor()

        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self.lab.max_workers,
            initializer=init_task_worker,
            initargs=(self.log_queue,),
        )

    def get_pbar(self, *, task_type: Type[Task], task_count: int) -> tqdm:
        pbar_func = tqdm_notebook if self.lab.notebook else tqdm
        return pbar_func(
            desc=task_type.__qualname__,
            unit='tasks',
            total=task_count,
            bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}] |{bar}| {percentage:3.0f}% [{remaining} remaining]',
            disable=self.disable_progress,
        )

    def handle_failure(self, *, ex: Exception, message: str):
        # Simplify subprocess error tracebacks by reporting the cause directly.
        if isinstance(ex.__cause__, concurrent.futures.process._RemoteTraceback):
            ex = ex.__cause__

        if self.lab.continue_on_failure:
            logger.error(f"{message} Skipping: {ex}")
        else:
            logger.error(message)

        lab_error = LabError(str(ex))
        if _IN_TASK_SUBPROCESS:
            # Simplify traceback by clearing the exception chain.
            raise lab_error from None
        raise lab_error from ex

    def use_cache(self, task: Task) -> bool:
        return (not self.bust_cache) and self.lab.is_cached(task)

    def run_or_load_task(self, process_name: str, task: Task) -> TaskResult:
        orig_process_name = multiprocessing.current_process().name
        try:
            multiprocessing.current_process().name = process_name
            if self.use_cache(task):
                logger.debug(f"Loading from cache: '{task}'")
                task_result = task._lt.cache.load_result_with_meta(self.lab._storage, task)
                return task_result
            else:
                logger.debug(f"Running: '{task}'")
                task.set_context(self.lab.context)
                with optional_mlflow(task):
                    start = datetime.now()
                    result = task.run()
                    end = datetime.now()
                task_result = TaskResult(
                    value=result,
                    meta=ResultMeta(
                        start=start,
                        duration=(end - start),
                    ),
                )
                task._lt.cache.save(self.lab._storage, task, task_result)
                logger.debug(f"Completed: '{task}'")
                return task_result
        finally:
            multiprocessing.current_process().name = orig_process_name

    def logger_thread(self):
        # See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        while True:
            record = self.log_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    def run(self, tasks: Sequence[Task]) -> Dict[Task, Any]:
        task_results = {}

        log_thread = Thread(target=self.logger_thread)
        log_thread.start()

        state = TaskState(runner=self, tasks=tasks)
        task_type_counts = Counter([type(task) for task in state.pending_tasks])
        task_type_max_digits = {
            task_type: math.ceil(math.log10(task_type_counts[task_type]))
            for task_type, task_type_count
            in task_type_counts.items()
        }
        task_type_process_counts = {task_type: 0 for task_type in task_type_counts.keys()}
        pbars = {
            task_type: self.get_pbar(
                task_type=task_type,
                task_count=task_count,
            )
            for task_type, task_count in task_type_counts.items()
        }
        executor = self.get_executor()
        serial_executor = self.get_executor(serial=True)
        future_to_task: Dict[concurrent.futures.Future, Task] = {}
        redirected_loggers = [] if self.lab.notebook else [logger]
        with logging_redirect_tqdm(loggers=redirected_loggers):
            try:
                try:
                    while (len(state.pending_tasks) > 0) or (len(future_to_task) > 0):
                        ready_tasks = state.get_ready_tasks()
                        for task in ready_tasks:
                            state.start_task(task)
                            task_type_process_counts[type(task)] += 1
                            process_id = (str(task_type_process_counts[type(task)])
                                          .zfill(task_type_max_digits[type(task)]))
                            process_name = f'{type(task).__name__}[{process_id}]'
                            # Always use a serial executor to run tasks on the
                            # main process if max_parallel=1
                            future_executor = (
                                serial_executor
                                if task._lt.max_parallel is not None and task._lt.max_parallel == 1
                                else executor
                            )
                            future = future_executor.submit(self.run_or_load_task, process_name, task)
                            future_to_task[future] = task
                        done, _ = wait_for_first_future(list(future_to_task.keys()))
                        for future in done:
                            task = future_to_task[future]
                            try:
                                task_result = future.result()
                            except Exception as ex:
                                state.complete_task(task, task_result=None)
                                self.handle_failure(ex=ex, message=f"Task '{task}' failed.")
                            else:
                                if task in tasks:
                                    task_results[task] = task_result.value
                                state.complete_task(task, task_result=task_result)
                                pbars[type(task)].update(1)
                        future_to_task = {future: future_to_task[future]
                                          for future in future_to_task
                                          if future not in done}
                except KeyboardInterrupt:
                    logger.info(('Interrupted. Finishing running tasks. '
                                 'Press Ctrl-C again to terminate running tasks immediately.'))
                else:
                    return task_results
                finally:
                    for future in future_to_task:
                        future.cancel()
                    executor.shutdown(wait=True)
            except KeyboardInterrupt:
                logger.info('Terminating running tasks.')
                for process in executor._processes.values():
                    process.terminate()
                executor.shutdown(wait=False)
            finally:
                for pbar in pbars.values():
                    pbar.close()
                self.log_queue.put(None)
                log_thread.join()

            raise KeyboardInterrupt


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
                 storage: Union[str, Path, None, Storage],
                 continue_on_failure: bool = True,
                 max_workers: Optional[int] = None,
                 notebook: bool = False,
                 context: Optional[dict[str, Any]] = None):
        """
        Args:
            storage: Where task results should be cached to. A string or
                [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
                will be interpreted as the path to a local directory, `None`
                will result in no caching. Any [Storage][labtech.types.Storage]
                instance may also be specified.
            continue_on_failure: If `True`, exceptions raised by tasks will be
                logged, but execution of other tasks will continue.
            max_workers: The maximum number of parallel worker processes for
                running tasks. Uses the same default as
                `concurrent.futures.ProcessPoolExecutor`: the number of
                processors on the machine. When `max_workers=1`, all tasks will
                be run in the main process, without multi-processing.
            notebook: Should be set to `True` if run from a Jupyter notebook
                for graphical progress bars.
            context: A dictionary of additional variables to make available to
                tasks. The context will not be cached, so the values should not
                affect results (e.g. parallelism factors) or should be kept
                constant between runs (e.g. datasets).
        """
        if isinstance(storage, str) or isinstance(storage, Path):
            storage = LocalStorage(storage)
        elif storage is None:
            storage = NullStorage()
        self._storage = storage
        self.continue_on_failure = continue_on_failure
        self.max_workers = max_workers
        self.notebook = notebook
        if context is None:
            context = {}
        self.context = context

    def run_tasks(self, tasks: Sequence[TaskT], *,
                  bust_cache: bool = False,
                  keep_nested_results: bool = False,
                  disable_progress: bool = False) -> Dict[TaskT, Any]:
        """Run the given tasks with as much process parallelism as possible.
        Loads task results from the cache storage where possible and
        caches results of executed tasks.

        Any attribute of a task that is itself a task object is
        considered a "nested task", and will be executed or loaded so
        that it's result is made available to its parent task. If the
        same task is nested inside multiple task objects, it will only
        be executed/loaded once.

        As well as returning the results, each task's result will be
        assigned to a `result` attribute on the task itself (including
        nested tasks when `keep_nested_results` is `True`).

        Args:
            tasks: The tasks to execute. Each should be an instance of a class
                decorated with [`labtech.task`][labtech.task].
            bust_cache: If `True`, no task results will be loaded from the
                cache storage; all tasks will be re-executed.
            keep_nested_results: If `False`, results of nested tasks that were
                executed or loaded in order to complete the provided tasks will
                be cleared from memory once they are no longer needed.
            disable_progress: If `True`, do not display a tqdm progress bar
                tracking task execution.

        Returns:
            A dictionary mapping each of the provided tasks to its
                corresponding result.

        """
        runner = TaskRunner(self,
                            bust_cache=bust_cache,
                            keep_nested_results=keep_nested_results,
                            disable_progress=disable_progress)
        results = runner.run(tasks)
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

    def cached_tasks(self, task_types: Sequence[Type[TaskT]]) -> Sequence[TaskT]:
        """Returns all task instances present in the Lab's cache storage for
        the given `task_types`, each of which should be a task class
        decorated with [`labtech.task`][labtech.task].

        Does not load task results from the cache storage, but they
        can be loaded by calling
        [`run_tasks()`][labtech.Lab.run_tasks] with the returned task
        instances.

        """
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
        return task._lt.cache.is_cached(self._storage, task)

    def uncache_tasks(self, tasks: Sequence[Task]):
        """Removes cached results for the given tasks from the Lab's cache
        storage."""
        for task in tasks:
            if self.is_cached(task):
                task._lt.cache.delete(self._storage, task)
