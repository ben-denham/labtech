import labtech
import mlflow


@labtech.task(mlflow_run=True)
class Run:
    seed: int

    def run(self):
        labtech.logger.info(f'Running with seed {self.seed}')
        return self.seed ** self.seed


runs = [
    Run(
        seed=seed,
    )
    for seed in range(10)
]

mlflow.set_tracking_uri("examples/storage/mlruns")
mlflow.set_experiment('example_labtech_experiment')
lab = labtech.Lab(
    storage=None,
)
results = lab.run_tasks(runs)
print(results)
