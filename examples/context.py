import labtech

datasets = {
    'primes': [1, 2, 3, 5, 7, 11],
    'fibonacci': [1, 1, 2, 3, 5, 8],
}


@labtech.task
class Experiment:
    dataset_key: str

    def run(self):
        labtech.logger.info(f'Running with dataset: {self.dataset_key}')
        datasets = self.context['datasets']
        return sum(datasets[self.dataset_key])


def main():
    experiments = [
        Experiment(
            dataset_key=dataset_key
        )
        for dataset_key in datasets.keys()
    ]

    lab = labtech.Lab(
        storage='examples/storage/context_lab',
        context={
            'datasets': datasets,
        },
    )
    cached_experiments = lab.cached_tasks([Experiment])
    print(f'Clearing {len(cached_experiments)} cached experiments.')
    lab.uncache_tasks(cached_experiments)
    results = lab.run_tasks(experiments)
    print(results)


if __name__ == '__main__':
    main()
