from __future__ import annotations

import functools
import multiprocessing
import os
from dataclasses import dataclass, field
from enum import StrEnum, auto
from itertools import count
from queue import Empty
from threading import Thread
from typing import TYPE_CHECKING

from labtech.exceptions import TaskDiedError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from multiprocessing.context import BaseContext
    from queue import Queue
    from typing import Any


class FutureStateError(Exception):
    pass


class FutureState(StrEnum):
    PENDING = auto()
    CANCELLED = auto()
    FINISHED = auto()


@dataclass
class ExecutorFuture:
    """Representation of a result to be returned in the future by a runner.

    An ExecutorFuture's state transitions between states according to
    the following finite state machine:

    * An ExecutorFuture starts in a PENDING state
    * A PENDING ExecutorFuture can be transitioned to FINISHED by calling
      set_result() or set_exception()
    * Any ExecutorFuture can be transitioned to CANCELLED by calling
      cancel()
    * result() can only be called on a FINISHED ExecutorFuture, and it will
      either return the result set by set_result() or raise the exception
      set by set_exception()

    """
    # Auto-incrementing ID (does not need to be process-safe because
    # all futures are generated in the main process):
    id: int = field(default_factory=count().__next__, init=False)
    _state: FutureState = FutureState.PENDING
    _ex: BaseException | None = None
    _result: Any | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return id(self.id) == id(other.id)

    def __hash__(self) -> int:
        return hash(self.id)

    def done(self) -> bool:
        return self._state in {FutureState.FINISHED, FutureState.CANCELLED}

    def cancelled(self) -> bool:
        return self._state == FutureState.CANCELLED

    def set_result(self, result: Any):
        if self.done():
            raise FutureStateError(f'Attempted to set a result on a {self._state} future.')
        self._result = result
        self._state = FutureState.FINISHED

    def set_exception(self, ex: BaseException):
        if self.done():
            raise FutureStateError(f'Attempted to set an exception on a {self._state} future.')
        self._ex = ex
        self._state = FutureState.FINISHED

    def cancel(self):
        self._state = FutureState.CANCELLED

    def result(self) -> Any:
        if self._state != FutureState.FINISHED:
            raise FutureStateError(f'Attempted to get result from a {self._state} future.')
        if self._ex is not None:
            raise self._ex
        return self._result


def split_done_futures(futures: Sequence[ExecutorFuture]) -> tuple[list[ExecutorFuture], list[ExecutorFuture]]:
    done_futures = []
    not_done_futures = []
    for future in futures:
        if future.done():
            done_futures.append(future)
        else:
            not_done_futures.append(future)
    return (done_futures, not_done_futures)


def _subprocess_target(*, future_id: int, thunk: Callable[[], Any], result_queue: Queue) -> None:
    try:
        result = thunk()
    except BaseException as ex:
        result_queue.put((future_id, ex))
    else:
        result_queue.put((future_id, result))


class ProcessExecutor:

    def __init__(self, mp_context: BaseContext, max_workers: int | None):
        self.mp_context = mp_context
        self.max_workers = (os.cpu_count() or 1) if max_workers is None else max_workers
        self._pending_future_to_thunk: dict[ExecutorFuture, Callable[[], Any]] = {}
        self._running_id_to_future_and_process: dict[int, tuple[ExecutorFuture, multiprocessing.Process]] = {}
        # Use a Manager().Queue() to be able to share with subprocesses
        self._result_queue: Queue = multiprocessing.Manager().Queue(-1)

    def _start_processes(self):
        """Start processes for the oldest pending futures to bring
        running process count up to max_workers."""
        start_count = max(0, self.max_workers - len(self._running_id_to_future_and_process))
        futures_to_start = list(self._pending_future_to_thunk.keys())[:start_count]
        for future in futures_to_start:
            thunk = self._pending_future_to_thunk[future]
            del self._pending_future_to_thunk[future]
            process = self.mp_context.Process(
                target=_subprocess_target,
                kwargs=dict(
                    future_id=future.id,
                    thunk=thunk,
                    result_queue=self._result_queue,
                ),
            )
            self._running_id_to_future_and_process[future.id] = (future, process)
            process.start()

    def submit(self, fn: Callable, /, *args, **kwargs) -> ExecutorFuture:
        """Schedule the given fn to be called with the given *args and
        **kwargs, and return an ExecutorFuture that will be updated
        with the outcome of function call."""
        future = ExecutorFuture()
        self._pending_future_to_thunk[future] = functools.partial(fn, *args, **kwargs)
        self._start_processes()
        return future

    def cancel(self) -> None:
        """Cancel all pending futures."""
        pending_futures = list(self._pending_future_to_thunk.keys())
        for future in pending_futures:
            future.cancel()
            del self._pending_future_to_thunk[future]

    def stop(self) -> None:
        """Cancel all running futures and immediately terminate their execution."""
        future_process_pairs = list(self._running_id_to_future_and_process.values())
        for future, process in future_process_pairs:
            process.terminate()
            future.cancel()
            del self._running_id_to_future_and_process[future.id]

    def _consume_result_queue(self, *, timeout_seconds: float | None):
        # Avoid race condition of a process finishing after we have
        # consumed the result_queue by fetching process statuses
        # before checking for process completion.
        dead_process_futures = [
            future for future, process in self._running_id_to_future_and_process.values()
            if not process.is_alive()
        ]

        def _consume():
            inner_timeout_seconds = timeout_seconds
            while True:
                try:
                    future_id, result_or_ex = self._result_queue.get(True, timeout=inner_timeout_seconds)
                except Empty:
                    break

                # Don't wait for the timeout on subsequent calls to
                # self._result_queue.get()
                inner_timeout_seconds = 0

                future, _ = self._running_id_to_future_and_process[future_id]
                del self._running_id_to_future_and_process[future_id]
                if not future.done():
                    if isinstance(result_or_ex, BaseException):
                        future.set_exception(result_or_ex)
                    else:
                        future.set_result(result_or_ex)

        # Consume the result queue in a thread so that it is not
        # interrupt by KeyboardInterrupt, which can result in us not
        # fully processing a completed result. Despite the fact we are
        # using subprocesses, starting a thread at this point should
        # be safe because we will not start any subprocesses while
        # this is running?
        consumer_thread = Thread(target=_consume)
        consumer_thread.start()
        consumer_thread.join()

        # If any processes have died without the future being
        # cancelled or finished, then set an exception for it.
        for future in dead_process_futures:
            if future.done():
                continue
            future.set_exception(TaskDiedError())
            del self._running_id_to_future_and_process[future.id]

    def wait(self, futures: Sequence[ExecutorFuture], *, timeout_seconds: float | None) -> tuple[list[ExecutorFuture], list[ExecutorFuture]]:
        """Wait up to timeout_seconds or until at least one of the
        given futures is done, then return a list of futures in a done
        state and a list of futures in all other states."""
        self._consume_result_queue(timeout_seconds=timeout_seconds)
        # Having consumed completed results, start new processes
        self._start_processes()
        return split_done_futures(futures)
