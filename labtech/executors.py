from concurrent.futures import Executor, Future, FIRST_COMPLETED, wait
from typing import Callable, Sequence


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
