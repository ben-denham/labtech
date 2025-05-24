from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from itertools import zip_longest
from string import Template
from typing import TYPE_CHECKING, cast

import psutil

from .exceptions import LabError
from .utils import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .types import Runner, TaskMonitorInfo, TaskMonitorInfoItem, TaskMonitorInfoValue


def get_info_value(task_info_item: TaskMonitorInfoItem) -> TaskMonitorInfoValue:
    if isinstance(task_info_item, tuple):
        return task_info_item[0]
    return task_info_item


def get_info_formatted(task_info_item: TaskMonitorInfoItem) -> str:
    if isinstance(task_info_item, tuple):
        return task_info_item[1]
    return str(task_info_item)


class MultilineDisplay(ABC):

    @abstractmethod
    def update(self, lines: Sequence[str]) -> None:
        pass

    @abstractmethod
    def show(self) -> None:
        pass

    def close(self) -> None:
        pass


class TerminalMultilineDisplay(MultilineDisplay):

    def __init__(self, *, line_count: int):
        self.pbars = [
            tqdm(
                bar_format='{desc}\033[K',
                dynamic_ncols=True,
            )
            for _ in range(line_count)
        ]

    def update(self, lines: Sequence[str]) -> None:
        if len(lines) > len(self.pbars):
            raise LabError(f'Multiline display updated with too many lines ({len(lines)}). This should not happen.')
        for pbar, line in zip_longest(self.pbars, lines, fillvalue=''):
            # Safe to cast pbar, as pbars will always be the longest
            # list and items will never be strings.
            cast('tqdm', pbar).set_description_str(line)

    def show(self) -> None:
        pass

    def close(self) -> None:
        for pbar in self.pbars:
            pbar.close()


class NotebookMultilineDisplay(MultilineDisplay):

    def __init__(self):
        from ipywidgets import Output
        self.output = Output()

    def update(self, lines: Sequence[str]) -> None:
        self.output.clear_output(wait=True)
        with self.output:
            print('\n'.join(lines), flush=True)

    def show(self) -> None:
        from IPython.display import display
        display(self.output)


class TaskMonitor:

    def __init__(self, *, runner: Runner, notebook: bool,
                 top_format: str, top_sort: str, top_n: int):
        self.runner = runner
        self.top_template = Template(top_format)
        self.top_sort = top_sort
        self.top_sort_key = top_sort
        self.top_sort_reversed = False
        if self.top_sort_key.startswith('-'):
            self.top_sort_key = self.top_sort_key[1:]
            self.top_sort_reversed = True
        self.top_n = top_n
        self.display = (
            NotebookMultilineDisplay() if notebook
            else TerminalMultilineDisplay(line_count=(top_n + 1))
        )

    def _top_task_lines(self) -> tuple[int, list[str]]:
        task_infos = [
            # Make (shallow) copies of dictionaries to avoid mutating
            # original dictionaries provided by runner.
            info.copy()
            for info in self.runner.get_task_infos()
        ]
        total_task_count = len(task_infos)

        # Sort order
        task_infos = sorted(task_infos, key=lambda info: get_info_value(info[self.top_sort_key]))
        if self.top_sort_reversed:
            task_infos = list(reversed(task_infos))
        # Take top
        task_infos = task_infos[:self.top_n]

        if len(task_infos) == 0:
            return total_task_count, []

        # Pad keys to consistent lengths
        for key, item in task_infos[0].items():
            # Left-align keys that contain string values
            left_align = isinstance(get_info_value(item), str)
            max_len = max([len(get_info_formatted(task_info[key])) for task_info in task_infos])
            for task_info in task_infos:
                align = '<' if left_align else '>'
                task_info[key] = (f'{{:{align}{max_len}}}').format(get_info_formatted(task_info[key]))

        # Final templating
        top_task_lines = [
            self.top_template.substitute(task_info)
            for task_info in task_infos
        ]
        return total_task_count, top_task_lines

    def update(self) -> None:
        """Called to update the monitor's displayed content."""
        total_task_count, top_task_lines = self._top_task_lines()
        self.display.update([
            (f'{total_task_count} active tasks '
             f'as at {datetime.now().strftime("%H:%M:%S")}. '
             f'Up to top {self.top_n} by {self.top_sort}:'),
            *top_task_lines,
        ])

    def close(self) -> None:
        self.display.close()

    def show(self) -> None:
        self.display.show()


def get_process_info(process: psutil.Process, *,
                     previous_child_processes: dict[int, psutil.Process],
                     name: str,
                     status: str) -> tuple[TaskMonitorInfo | None, dict[int, psutil.Process]]:
    """Utility for constructing a TaskMonitorInfo for a given process.

    Because psutil reports 0% CPU usage for newly created process
    objects, this function receives a previously instantiated process
    and a dictionary of child process objects (mapping pid ->
    process). The latest child processes are returned by this function
    to be re-used in the next invocation for the same process.

    """
    try:
        with process.oneshot():
            start_datetime = datetime.fromtimestamp(process.create_time())
            threads = process.num_threads()
            cpu_percent = process.cpu_percent()
            memory_rss_percent = process.memory_percent('rss')
            memory_vms_percent = process.memory_percent('vms')
            latest_child_process_list = process.children(recursive=True)
    except psutil.NoSuchProcess:
        return None, {}

    child_processes = {
        # Re-use previous child process object where available.
        child.pid: previous_child_processes.get(child.pid, child)
        for child in latest_child_process_list
    }
    for child in child_processes.values():
        try:
            with child.oneshot():
                threads += child.num_threads()
                cpu_percent += child.cpu_percent()
                memory_rss_percent += child.memory_percent('rss')
                memory_vms_percent += child.memory_percent('vms')
        except psutil.NoSuchProcess:
            # Ignore dead child processes
            continue

    info: TaskMonitorInfo = {
        'pid': process.pid,
        'name': name,
        'status': status,
        'start_time': (start_datetime, start_datetime.strftime('%H:%M:%S')),
        'children': len(child_processes),
        'threads': threads,
        'cpu': (cpu_percent, f'{cpu_percent/100:.1%}'),
        'rss': (memory_rss_percent, f'{memory_rss_percent/100:.1%}'),
        'vms': (memory_vms_percent, f'{memory_vms_percent/100:.1%}'),
    }
    return info, child_processes
