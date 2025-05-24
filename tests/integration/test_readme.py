import subprocess
import sys
from pathlib import Path

ROOT_PROJ_DIR = Path(__file__).parents[3]
README_PKG = Path(__file__).parent / 'readme'


class TestReadmeExamples:
    def test_usage_subprocess(self) -> None:
        cproc = _run_example_subprocess('usage')
        assert cproc.stdout.decode().replace('\r', '') == '[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 3, 9, 27, 81, 1, 4, 16, 64, 256]\n'

    def test_dependents_and_mermaid_subprocess(self) -> None:
        cproc = _run_example_subprocess('dependents_and_mermaid')
        listout = '[0, 1764, 3528, 5292, 7056, 8820, 10584, 12348, 14112, 15876]\n'
        diagram = (
            'classDiagram\n'
            '    direction BT\n'
            '\n'
            '    class DependentTask\n'
            '    DependentTask : SlowTask slow_task\n'
            '    DependentTask : int multiplier\n'
            '    DependentTask : run() int\n'
            '\n'
            '    class SlowTask\n'
            '    SlowTask : int base\n'
            '    SlowTask : run() int\n'
            '\n'
            '\n'
            '    DependentTask <-- SlowTask: slow_task\n'
        )

        stdout = cproc.stdout.decode().replace('\r', '')
        assert stdout == (listout + diagram)


def _run_example_subprocess(name: str) -> subprocess.CompletedProcess:
    cproc = subprocess.run(
        [
            sys.executable,
            (README_PKG / f'{name}.py').relative_to(ROOT_PROJ_DIR).as_posix(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT_PROJ_DIR,
    )
    msg = cproc.stderr.decode() if cproc.stderr else None
    assert (cproc.returncode == 0), msg
    return cproc
