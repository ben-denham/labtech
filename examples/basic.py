from enum import Enum

import labtech


class Multiplier(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


@labtech.task
class Experiment:
    seed: int
    multiplier: Multiplier

    def run(self):
        labtech.logger.info(f'Running with seed {self.seed} and multiplier {self.multiplier}')
        return self.seed * self.multiplier.value


experiments = [
    Experiment(
        seed=seed,
        multiplier=multiplier,
    )
    for seed in range(10)
    for multiplier in Multiplier
]


lab = labtech.Lab(
    storage='examples/storage/basic_lab',
)
cached_experiments = lab.cached_tasks([Experiment])
print(f'Clearing {len(cached_experiments)} cached experiments.')
lab.uncache_tasks(cached_experiments)
results = lab.run_tasks(experiments)
print(results)
