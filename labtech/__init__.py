"""Library to run experiment permutations with multi-processing and caching.

Typical usage:

```python
import labtech

@labtech.task
class Experiment:
    seed: int

    def run(self):
        labtech.logger.info(f'Running with seed {self.seed}')
        return 42 * self.seed

experiments = [
    Experiment(
        seed=seed,
    )
    for seed in range(10)
]


lab = labtech.Lab(
    storage='my_lab_storage_dir',
)
results = lab.run_tasks(experiments)
print(results)

```

"""

__version__ = '0.8.1'

from .lab import Lab
from .tasks import task
from .types import is_task, is_task_type
from .utils import logger

__all__ = [
    'is_task_type',
    'is_task',
    'task',
    'Lab',
    'logger',
]
