import labtech


def test_duplicated_task():
    """Tests both duplicate parent tasks and duplicated child tasks."""

    @labtech.task(cache=None)
    class ChildTask:

        def run(self):
            return 1

    @labtech.task(cache=None)
    class ParentTask:
        child: ChildTask

        def run(self):
            return self.child.result

    tasks = [
        ParentTask(child=ChildTask()),
        ParentTask(child=ChildTask()),
    ]

    lab = labtech.Lab(storage=None, max_workers=1)
    results = lab.run_tasks(tasks)

    assert results == {
        ParentTask(child=ChildTask()): 1,
    }

    for task in tasks:
        assert task.result_meta is not None
        assert task.child.result_meta is not None
