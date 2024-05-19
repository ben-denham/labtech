from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from queue import Empty, Queue
from string import Template
from threading import Timer
from typing import Any, Dict, List, Optional, Sequence, cast

import psutil
from tqdm import tqdm

from .exceptions import LabError


class TaskEvent:
    pass


@dataclass(frozen=True)
class TaskStartEvent(TaskEvent):
    process_name: str
    pid: int
    use_cache: bool


@dataclass(frozen=True)
class TaskEndEvent(TaskEvent):
    process_name: str


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

    def __init__(self, *, task_monitor_queue: Queue, notebook: bool,
                 top_format: str, top_sort: str, top_n: int,
                 update_interval_seconds: float = 0.5):
        self.task_monitor_queue = task_monitor_queue
        self.top_template = Template(top_format)
        self.top_sort = top_sort
        self.top_sort_key = top_sort
        self.top_sort_reversed = False
        if self.top_sort_key.startswith('-'):
            self.top_sort_key = self.top_sort_key[1:]
            self.top_sort_reversed = True
        self.top_n = top_n
        self.update_interval_seconds = update_interval_seconds
        self.display = (
            NotebookMultilineDisplay() if notebook
            else TerminalMultilineDisplay(line_count=(top_n + 1))
        )
        self.active_task_events: Dict[str, TaskStartEvent] = {}
        self.timer: Optional[Timer] = None
        self.stopped = True

    def _consume_monitor_queue(self):
        while True:
            try:
                event = self.task_monitor_queue.get_nowait()
            except Empty:
                break

            if isinstance(event, TaskStartEvent):
                self.active_task_events[event.process_name] = event
            elif isinstance(event, TaskEndEvent):
                if event.process_name in self.active_task_events:
                    del self.active_task_events[event.process_name]
            else:
                raise LabError(f'Unexpected task event: {event}')

    def _get_process_info(self, start_event: TaskStartEvent) -> Optional[dict[str, Any]]:
        pid = start_event.pid
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return None
        with process.oneshot():
            start_datetime = datetime.fromtimestamp(process.create_time())
            threads = process.num_threads()
            cpu_percent = process.cpu_percent()
            memory_rss_percent = process.memory_percent('rss')
            memory_vms_percent = process.memory_percent('vms')
            children = process.children(recursive=True)
        for child in children:
            with child.oneshot():
                threads += child.num_threads()
                cpu_percent += child.cpu_percent()
                memory_rss_percent += child.memory_percent('rss')
                memory_vms_percent += child.memory_percent('vms')
        return {
            'name': start_event.process_name,
            'pid': pid,
            'status': ('loading' if start_event.use_cache else 'running'),
            'start_time': start_datetime,
            'children': len(children),
            'threads': threads,
            'cpu': cpu_percent,
            'rss': memory_rss_percent,
            'vms': memory_vms_percent,
        }

    def _top_task_lines(self) -> List[str]:
        process_infos: List[dict[str, Any]] = []
        for start_event in self.active_task_events.values():
            process_info = self._get_process_info(start_event)
            if process_info is not None:
                process_infos.append(process_info)
        # Sort order
        process_infos = sorted(process_infos, key=lambda info: info[self.top_sort_key])
        if self.top_sort_reversed:
            process_infos = list(reversed(process_infos))
        # Take top
        process_infos = process_infos[:self.top_n]
        # Value formatting
        for process_info in process_infos:
            process_info['start_time'] = process_info['start_time'].strftime('%H:%M:%S')
            process_info['cpu'] = f'{process_info["cpu"]/100:.1%}'
            process_info['rss'] = f'{process_info["rss"]/100:.1%}'
            process_info['vms'] = f'{process_info["vms"]/100:.1%}'

        if len(process_infos) == 0:
            return []

        # Pad keys to consistent lengths
        left_align_keys = {'name', 'status'}
        for key in process_infos[0].keys():
            max_len = max([len(str(process_info[key])) for process_info in process_infos])
            for process_info in process_infos:
                align = '<' if key in left_align_keys else '>'
                process_info[key] = (f'{{:{align}{max_len}}}').format(process_info[key])

        # Final templating
        return [
            self.top_template.substitute(process_info)
            for process_info in process_infos
        ]

    def _update(self) -> None:
        self._consume_monitor_queue()
        # Update display
        self.display.update([
            (f'{len(self.active_task_events)} active tasks '
             f'as at {datetime.now().strftime("%H:%M:%S")}. '
             f'Up to top {self.top_n} by {self.top_sort}:'),
            *self._top_task_lines(),
        ])
        # Schedule next update
        if not self.stopped:
            self.timer = Timer(self.update_interval_seconds, self._update)
            self.timer.start()

    def start(self) -> None:
        self.stopped = False
        self._update()

    def close(self) -> None:
        self.stopped = True
        if self.timer is not None:
            self.timer.join()
        self.display.close()

    def show(self) -> None:
        self.display.show()
