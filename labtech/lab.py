"""Lab and related utilities responsible for running tasks."""

from collections import Counter, defaultdict
import concurrent.futures
from dataclasses import fields
import logging
from logging.handlers import QueueHandler
import math
import multiprocessing
import signal
import sys
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Type, Union

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.contrib.logging import logging_redirect_tqdm

from .types import Task, ResultsMap, Storage, is_task
from .exceptions import LabError, TaskNotFound
from .utils import OrderedSet, LoggerFileProxy, logger
from .storage import NullStorage, LocalStorage


class TaskState:

    def __init__(self, *, runner: 'TaskRunner', tasks: Sequence[Task]):
        self.runner = runner
        self.results_map: ResultsMap = {}

        self.pending_tasks: OrderedSet[Task] = OrderedSet()
        self.task_to_direct_dependencies: Dict[Task, Set[Task]] = defaultdict(set)
        self.task_to_pending_dependencies: Dict[Task, Set[Task]] = defaultdict(set)
        self.task_to_pending_dependents: Dict[Task, Set[Task]] = defaultdict(set)
        self.type_to_active_tasks: Dict[Type[Task], Set[Task]] = defaultdict(set)

        self.process_tasks(tasks)
        self.check_cyclic_dependences()

    def find_tasks(self, value, searched_coll_ids: Optional[Set[int]] = None) -> Sequence[Task]:
        if searched_coll_ids is None:
            searched_coll_ids = set()
        if id(value) in searched_coll_ids:
            return []

        if is_task(value):
            return [value]
        elif isinstance(value, list) or isinstance(value, tuple):
            searched_coll_ids = searched_coll_ids | {id(value)}
            return [
                task
                for item in value
                for task in self.find_tasks(item, searched_coll_ids)
            ]
        elif isinstance(value, dict):
            searched_coll_ids = searched_coll_ids | {id(value)}
            return [
                task
                for item in value.values()
                for task in self.find_tasks(item, searched_coll_ids)
            ]
        return []

    def process_tasks(self, tasks: Iterable[Task]):
        task_dependencies = {}
        for task in tasks:
            dependent_tasks: List[Task] = []
            if not self.runner.use_cache(task):
                # Find all dependent tasks inside task fields
                for field in fields(task):
                    field_value = getattr(task, field.name)
                    dependent_tasks += self.find_tasks(field_value)
            task_dependencies[task] = dependent_tasks
        # We insert all of the top-level tasks before processing
        # discovered dependencies, so that we will attempt to run
        # higher-level tasks as soon as possible.
        for task, dependencies in task_dependencies.items():
            self.insert_task(task, dependencies)
        all_dependencies: List[Task] = sum(task_dependencies.values(), start=[])
        if len(all_dependencies) > 0:
            self.process_tasks(all_dependencies)

    def insert_task(self, task: Task, dependencies: Iterable[Task]):
        task._set_results_map(self.results_map)
        self.pending_tasks.add(task)
        for dependency in dependencies:
            self.task_to_direct_dependencies[task].add(dependency)
            self.task_to_pending_dependencies[task].add(dependency)
            self.task_to_pending_dependents[dependency].add(task)

    def start_task(self, task: Task):
        self.pending_tasks.remove(task)
        self.type_to_active_tasks[type(task)].add(task)

    def complete_task(self, task: Task, *, success: bool, result: Any):
        if success:
            self.results_map[task] = result

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


class TaskRunner:

    def __init__(self, lab: 'Lab', bust_cache: bool, keep_nested_results: bool,
                 disable_progress: bool):
        self.lab = lab
        self.bust_cache = bust_cache
        self.keep_nested_results = keep_nested_results
        self.disable_progress = disable_progress
        self.log_queue = multiprocessing.Manager().Queue(-1)

    def get_executor(self):

        def init_worker():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Sub-processes should log onto the queue in order to printed
            # in serial by the main process.
            logger.handlers = []
            logger.addHandler(QueueHandler(self.log_queue))
            sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')
            sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')

        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self.lab.max_workers,
            initializer=init_worker,
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
        # Simplify traceback by clearing the exception chain.
        raise LabError(str(ex)) from None

    def use_cache(self, task: Task):
        return (not self.bust_cache) and self.lab.is_cached(task)

    def run_or_load_task(self, process_name: str, task: Task):
        multiprocessing.current_process().name = process_name
        if self.use_cache(task):
            logger.debug(f"Loading from cache: '{task}'")
            return task._lt.cache.load_result(self.lab._storage, task)
        else:
            logger.debug(f"Running: '{task}'")
            result = task.run()
            task._lt.cache.save(self.lab._storage, task, result)
            logger.debug(f"Completed: '{task}'")
            return result

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
                            future = executor.submit(self.run_or_load_task, process_name, task)
                            future_to_task[future] = task
                        done, _ = concurrent.futures.wait(
                            future_to_task,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            task = future_to_task[future]
                            try:
                                result = future.result()
                            except Exception as ex:
                                state.complete_task(task, success=False, result=None)
                                self.handle_failure(ex=ex, message=f"Task '{task}' failed.")
                            else:
                                if task in tasks:
                                    task_results[task] = result
                                state.complete_task(task, success=True, result=result)
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
                 storage: Union[str, None, Storage],
                 continue_on_failure: bool = True,
                 max_workers: Optional[int] = None,
                 notebook: bool = False):
        """
        Args:
            storage: Where task results should be cached to. A string will be
                interpreted as the path to a local directory, `None` will result
                in no caching. Any [Storage][labtech.types.Storage] instance may
                also be specified.
            continue_on_failure: If `True`, exceptions raised by tasks will be
                logged, but execution of other tasks will continue.
            max_workers: The maximum number of parallel worker processes for
                running tasks. Uses the same default as
                `concurrent.futures.ProcessPoolExecutor`: the number of
                processors on the machine.
            notebook: Should be set to `True` if run from a Jupyter notebook
                for graphical progress bars.
        """
        if isinstance(storage, str):
            storage = LocalStorage(storage)
        elif storage is None:
            storage = NullStorage()
        self._storage = storage
        self.continue_on_failure = continue_on_failure
        self.max_workers = max_workers
        self.notebook = notebook

    def run_tasks(self, tasks: Sequence[Task], *,
                  bust_cache: bool = False,
                  keep_nested_results: bool = False,
                  disable_progress: bool = False) -> Dict[Task, Any]:
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
        return runner.run(tasks)

    def cached_tasks(self, task_types: Sequence[Type[Task]]) -> Sequence[Task]:
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
