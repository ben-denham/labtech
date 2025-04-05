from tempfile import TemporaryDirectory
from time import sleep

import labtech


@labtech.task
class SlowTask:
    base: int

    def run(self) -> int:
        # sleep(5)
        return self.base ** 2

@labtech.task
class DependentTask:
    slow_task: SlowTask
    multiplier: int

    def run(self) -> int:
        return self.multiplier * self.slow_task.result

def main(storage_dir):
    some_slow_task = SlowTask(base=42)
    dependent_tasks = [
        DependentTask(
            slow_task=some_slow_task,
            multiplier=multiplier,
        )
        for multiplier in range(10)
    ]

    lab = labtech.Lab(storage=storage_dir)
    results = lab.run_tasks(dependent_tasks)
    print([results[task] for task in dependent_tasks])

if __name__ == '__main__':
    with TemporaryDirectory() as storage_dir:
        main(storage_dir)

    from labtech.diagram import display_task_diagram

    some_slow_task = SlowTask(base=42)
    dependent_tasks = [
        DependentTask(
            slow_task=some_slow_task,
            multiplier=multiplier,
        )
        for multiplier in range(10)
    ]

    display_task_diagram(dependent_tasks)
