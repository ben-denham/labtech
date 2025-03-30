from abc import ABC, abstractmethod
from datetime import datetime
from itertools import zip_longest
from string import Template
from typing import Sequence, cast

from .exceptions import LabError
from .types import Runner, TaskMonitorInfoItem, TaskMonitorInfoValue
from .utils import tqdm


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
            cast(tqdm, pbar).set_description_str(line)

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

    def _top_task_lines(self) -> list[str]:
        task_infos = self.runner.get_task_infos()
        # Sort order
        task_infos = sorted(task_infos, key=lambda info: get_info_value(info[self.top_sort_key]))
        if self.top_sort_reversed:
            task_infos = list(reversed(task_infos))
        # Take top
        task_infos = task_infos[:self.top_n]

        if len(task_infos) == 0:
            return []

        # Pad keys to consistent lengths
        for key, item in task_infos[0].items():
            # Left-align keys that contain string values
            left_align = isinstance(get_info_value(item), str)
            max_len = max([len(get_info_formatted(task_info[key])) for task_info in task_infos])
            for task_info in task_infos:
                align = '<' if left_align else '>'
                task_info[key] = (f'{{:{align}{max_len}}}').format(get_info_formatted(task_info[key]))

        # Final templating
        return [
            self.top_template.substitute(task_info)
            for task_info in task_infos
        ]

    def update(self) -> None:
        """Called to update the monitor's displayed content."""
        top_task_lines = self._top_task_lines()
        self.display.update([
            (f'{len(top_task_lines)} active tasks '
             f'as at {datetime.now().strftime("%H:%M:%S")}. '
             f'Up to top {self.top_n} by {self.top_sort}:'),
            *top_task_lines,
        ])

    def close(self) -> None:
        self.display.close()

    def show(self) -> None:
        self.display.show()
