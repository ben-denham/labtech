import logging
import multiprocessing
import signal
import sys
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from logging.handlers import QueueHandler
from threading import Thread
from typing import Callable, Iterator, Optional, Sequence
from uuid import UUID, uuid4

from labtech.tasks import get_direct_dependencies
from labtech.types import LabContext, ResultMeta, ResultsMap, Runner, RunnerBackend, Storage, Task, TaskResult
from labtech.utils import LoggerFileProxy, logger

from .base import run_or_load_task


class SerialFuture(Future):

    def __init__(self, fn: Callable, /, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.set_running_or_notify_cancel():
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as ex:
            self.set_exception(ex)
        else:
            self.set_result(result)


class SerialExecutor(Executor):

    def __init__(self) -> None:
        self.futures: list[SerialFuture] = []

    def submit(self, fn: Callable, /, *args, **kwargs):
        future = SerialFuture(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        if cancel_futures:
            for future in self.futures:
                future.cancel()


def wait_for_first_future(futures: Sequence[Future]):
    # If there are any serial futures, ensure at least one is completed.
    serial_futures = [future for future in futures if isinstance(future, SerialFuture)]
    if serial_futures and not serial_futures[0].done():
        serial_futures[0].run()
    return wait(futures, return_when=FIRST_COMPLETED)


def init_task_subprocess(log_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Sub-processes should log onto the queue in order to printed
    # in serial by the main process.
    logger.handlers = []
    logger.addHandler(QueueHandler(log_queue))
    sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')
    sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')


class ProcessRunner(Runner, ABC):
    """TODO"""

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        self.log_queue = multiprocessing.Manager().Queue(-1)
        self.log_thread = Thread(target=self._logger_thread)
        self.log_thread.start()

        self.serial_executor = SerialExecutor()
        self.executor: Executor
        if max_workers is not None and max_workers == 1:
            self.executor = self.serial_executor
        else:
            self.executor = ProcessPoolExecutor(
                mp_context=self._get_mp_context(),
                max_workers=max_workers,
                initializer=init_task_subprocess,
                initargs=(self.log_queue,),
            )

        self.results_map: dict[Task, TaskResult] = {}
        self.future_to_task: dict[Future, Task] = {}
        self.closed = False

    def _logger_thread(self):
        # See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        while True:
            record = self.log_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    @staticmethod
    def _subprocess_func(*, task: Task, task_name: str, use_cache: bool,
                         results_map: ResultsMap, context: LabContext,
                         storage: Storage) -> TaskResult:
        orig_process_name = multiprocessing.current_process().name
        try:
            current_process = multiprocessing.current_process()
            current_process.name = task_name

            # if self.task_monitor_queue is not None:
            #     self.task_monitor_queue.put(TaskStartEvent(
            #         task_name=task_name,
            #         pid=cast(int, current_process.pid),
            #         use_cache=use_cache,
            #     ))

            for dependency_task in get_direct_dependencies(task):
                dependency_task._set_results_map(results_map)

            return run_or_load_task(
                task=task,
                task_name=task_name,
                use_cache=use_cache,
                context=context,
                storage=storage
            )
        finally:
            current_process.name = orig_process_name
            # if self.task_monitor_queue is not None:
            #     self.task_monitor_queue.put(TaskEndEvent(
            #         task_name=task_name,
            #     ))

    def start_task(self, task: Task, task_name: str, use_cache: bool) -> None:
        # Always use a serial executor to run tasks on the
        # main process if max_parallel=1
        executor = (
            self.serial_executor
            if task._lt.max_parallel is not None and task._lt.max_parallel == 1
            else self.executor
        )
        future = self._submit_task(
            executor=executor,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
        )
        self.future_to_task[future] = task

    def pending_task_count(self) -> int:
        return len(self.future_to_task)

    def wait(self) -> Iterator[tuple[Task, ResultMeta | Exception]]:
        done, _ = wait_for_first_future(list(self.future_to_task.keys()))
        for future in done:
            task = self.future_to_task[future]
            try:
                task_result = future.result()
            except Exception as ex:
                yield (task, ex)
            else:
                self.results_map[task] = task_result
                yield (task, task_result.meta)
        self.future_to_task = {
            future: self.future_to_task[future]
            for future in self.future_to_task
            if future not in done
        }

    def cancel(self) -> None:
        for future in self.future_to_task:
            future.cancel()

    def terminate(self) -> None:
        for process in self.executor._processes.values():  # type: ignore[attr-defined]
            process.terminate()
        self.close(wait=False)

    def close(self, *, wait: bool) -> None:
        if self.closed:
            return
        self.serial_executor.shutdown(wait=wait)
        self.executor.shutdown(wait=wait)
        self.log_queue.put(None)
        self.log_thread.join()

    def get_result(self, task: Task) -> TaskResult:
        return self.results_map[task]

    def remove_result(self, task: Task) -> None:
        if task not in self.results_map:
            return
        logger.debug(f"Removing result from in-memory cache for task: '{task}'")
        del self.results_map[task]

    @abstractmethod
    def _get_mp_context(self) -> multiprocessing.context.BaseContext:
        pass

    @abstractmethod
    def _submit_task(self, executor: Executor, task: Task, task_name: str, use_cache: bool) -> Future:
        # Should call _subprocess_func()
        pass


class SpawnProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.context = context
        self.storage = storage

    def _get_mp_context(self) -> multiprocessing.context.SpawnContext:
        return multiprocessing.get_context('spawn')

    def _submit_task(self, executor: Executor, task: Task, task_name: str, use_cache: bool) -> Future:
        return executor.submit(
            self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            results_map={
                dependency_task: self.results_map[dependency_task]
                for dependency_task in get_direct_dependencies(task)
            },
            # TODO: Allow a task to specify which context keys it needs
            context=self.context,
            storage=self.storage,
        )


class SpawnRunnerBackend(RunnerBackend):
    """TODO"""

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> SpawnProcessRunner:
        return SpawnProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )


@dataclass
class RunnerMemory:
    results_map: ResultsMap
    context: LabContext
    storage: Storage


_RUNNER_FORK_MEMORY: dict[UUID, RunnerMemory] = {}


class ForkProcessRunner(ProcessRunner):

    def __init__(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)
        self.uuid = uuid4()
        _RUNNER_FORK_MEMORY[self.uuid] = RunnerMemory(
            results_map=self.results_map,
            context=context,
            storage=storage,
        )

    def _get_mp_context(self) -> multiprocessing.context.ForkContext:
        return multiprocessing.get_context('fork')

    @staticmethod
    def _fork_subprocess_func(*, _subprocess_func: Callable, task: Task, task_name: str, use_cache: bool, uuid: UUID) -> TaskResult:
        runner_memory = _RUNNER_FORK_MEMORY[uuid]
        return _subprocess_func(
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            context=runner_memory.context,
            storage=runner_memory.storage,
            results_map=runner_memory.results_map,
        )

    def _submit_task(self, executor: Executor, task: Task, task_name: str, use_cache: bool) -> Future:
        return executor.submit(
            self._fork_subprocess_func,
            _subprocess_func=self._subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            uuid=self.uuid,
        )

    def close(self, *, wait: bool) -> None:
        del _RUNNER_FORK_MEMORY[self.uuid]
        super().close(wait=wait)


class ForkRunnerBackend(RunnerBackend):
    """TODO"""

    def build_runner(self, *, context: LabContext, storage: Storage, max_workers: Optional[int]) -> ForkProcessRunner:
        return ForkProcessRunner(
            context=context,
            storage=storage,
            max_workers=max_workers,
        )
