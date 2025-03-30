from tempfile import TemporaryDirectory
from time import sleep

import labtech


# Decorate your task class with @labtech.task:
@labtech.task
class Experiment:
    # Each Experiment task instance will take `base` and `power` parameters:
    base: int
    power: int

    def run(self) -> int:
        # Define the task's run() method to return the result of the experiment:
        labtech.logger.info(f'Raising {self.base} to the power of {self.power}')
        # sleep(1)
        return self.base ** self.power

def main(storage_dir):
    # Configure Experiment parameter permutations
    experiments = [
        Experiment(
            base=base,
            power=power,
        )
        for base in range(5)
        for power in range(5)
    ]

    # Configure a Lab to run the experiments:
    lab = labtech.Lab(
        # Specify a directory to cache results in (running the experiments a second
        # time will just load results from the cache!):
        storage=storage_dir,
        # Control the degree of parallelism:
        max_workers=5,
    )

    # Run the experiments!
    results = lab.run_tasks(experiments)
    print([results[experiment] for experiment in experiments])

if __name__ == '__main__':
    with TemporaryDirectory() as storage_dir:
        main(storage_dir)
