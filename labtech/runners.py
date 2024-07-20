import concurrent.futures
import logging
import multiprocessing
import signal
import sys
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime
from enum import Enum
from logging.handlers import QueueHandler
from threading import Thread
from typing import Any, Iterator, Optional

from frozendict import frozendict

from .exceptions import LabError
from .executors import SerialExecutor, wait_for_first_future
from .tasks import is_task
from .types import ResultMeta, Runner, Storage, Task, TaskResult
from .utils import LoggerFileProxy, logger


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


def run_or_load_task(task: Task, task_name: str, use_cache: bool, context: dict[str, Any], storage: Storage) -> TaskResult:
    if use_cache:
        logger.debug(f"Loading from cache: '{task}'")
        task_result = task._lt.cache.load_result_with_meta(storage, task)
        return task_result
    else:
        logger.debug(f"Running: '{task}'")
        task.set_context(context)
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
        task._lt.cache.save(storage, task, task_result)
        logger.debug(f"Completed: '{task}'")
        return task_result


class BaseRunner(Runner):

    def __init__(self, *, context: dict[str, Any], storage: Storage, max_workers: Optional[int]):
        self.context = context
        self.storage = storage
        self.max_workers = max_workers


def init_task_subprocess(log_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Sub-processes should log onto the queue in order to printed
    # in serial by the main process.
    logger.handlers = []
    logger.addHandler(QueueHandler(log_queue))
    sys.stdout = LoggerFileProxy(logger.info, 'Captured STDOUT:\n')
    sys.stderr = LoggerFileProxy(logger.error, 'Captured STDERR:\n')


class ProcessRunner(BaseRunner):
    """TODO"""

    def __init__(self, *, context: dict[str, Any], storage: Storage, max_workers: Optional[int]):
        super().__init__(context=context, storage=storage, max_workers=max_workers)

        self.log_queue = multiprocessing.Manager().Queue(-1)
        self.log_thread = Thread(target=self.logger_thread)
        self.log_thread.start()

        self.serial_executor = SerialExecutor()
        self.executor: concurrent.futures.Executor
        if max_workers is not None and max_workers == 1:
            self.executor = self.serial_executor
        else:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=init_task_subprocess,
                initargs=(self.log_queue,),
            )

        self.future_to_task: dict[concurrent.futures.Future, Task] = {}
        self.closed = False

    def logger_thread(self):
        # See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        while True:
            record = self.log_queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    @staticmethod
    def subprocess_func(*, task: Task, task_name: str, use_cache: bool, context: dict[str, Any], storage: Storage) -> TaskResult:
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

    def start_task(self, task: Task, task_name, use_cache: bool) -> None:
        # Always use a serial executor to run tasks on the
        # main process if max_parallel=1
        executor = (
            self.serial_executor
            if task._lt.max_parallel is not None and task._lt.max_parallel == 1
            else self.executor
        )
        future = executor.submit(
            self.subprocess_func,
            task=task,
            task_name=task_name,
            use_cache=use_cache,
            context=self.context,
            storage=self.storage,
        )
        self.future_to_task[future] = task

    def pending_task_count(self) -> int:
        return len(self.future_to_task)

    def wait_for_completed_tasks(self) -> Iterator[tuple[Task, TaskResult | Exception]]:
        done, _ = wait_for_first_future(list(self.future_to_task.keys()))
        for future in done:
            task = self.future_to_task[future]
            try:
                task_result = future.result()
            except Exception as ex:
                yield (task, ex)
            else:
                yield (task, task_result)
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

    def close(self, wait: bool) -> None:
        if self.closed:
            return
        self.serial_executor.shutdown(wait=wait)
        self.executor.shutdown(wait=wait)
        self.log_queue.put(None)
        self.log_thread.join()


class ForkProcessRunner(ProcessRunner):
    """TODO"""


class SpawnProcessRunner(Runner):
    """TODO"""
