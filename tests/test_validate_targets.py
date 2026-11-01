"""Pre-flight check of the goal targets before a multiobjective job starts."""

import pandas as pd
import pytest

from config import Config
from src.process import ExperimentRunner as runner
from src.process.RuntimeContext import validate_targets


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    data = tmp_path / "data" / "DATA_PRP_5C"
    data.mkdir(parents=True)
    names = ["PRP1_C5_P3_V1_T3_S1.dat", "PRP2_C5_P3_V1_T3_S1.dat"]
    for name in names:
        (data / name).write_text("x")
    (tmp_path / "targets").mkdir()

    def configure(multiobjective=True, build_target=False, experiments=None):
        monkeypatch.setattr(
            Config,
            "_data",
            Config.normalize(
                {
                    "solver": {"multiobjective": multiobjective, "method": "PSO"},
                    "instance": {
                        "dir": f"{tmp_path}/data/",
                        "output": f"{tmp_path}/out/",
                        "files": ["DATA_PRP_5C"],
                    },
                    "postprocessing": {
                        "build_target": build_target,
                        "output": f"{tmp_path}/targets/",
                    },
                    **({"experiments": experiments} if experiments else {}),
                }
            ),
        )

    def write(rows_by_file):
        rows = [
            {"file": f"{tmp_path}/data/DATA_PRP_5C/{name}", "time": t,
             **{f"f{j}_target": 1.0 for j in range(1, 6)}}
            for name, count in rows_by_file.items()
            for t in range(count)
        ]
        pd.DataFrame(rows).to_excel(tmp_path / "targets" / "targets.xlsx", index=False)

    return type("W", (), {"configure": staticmethod(configure), "write": staticmethod(write),
                          "names": names})


def test_complete_targets_pass(workspace):
    workspace.configure()
    workspace.write({name: 3 for name in workspace.names})

    validate_targets()


def test_missing_file_fails_before_any_task(workspace):
    workspace.configure()

    with pytest.raises(RuntimeError, match="ausente"):
        validate_targets()


def test_an_instance_without_all_periods_is_reported(workspace):
    workspace.configure()
    workspace.write({workspace.names[0]: 3, workspace.names[1]: 2})

    with pytest.raises(RuntimeError, match=workspace.names[1]):
        validate_targets()


def test_an_instance_without_targets_is_reported(workspace):
    workspace.configure()
    workspace.write({workspace.names[0]: 3})

    with pytest.raises(RuntimeError, match=workspace.names[1]):
        validate_targets()


@pytest.mark.parametrize("multiobjective, build_target", [(False, False), (True, True)])
def test_runs_that_do_not_use_targets_are_not_checked(workspace, multiobjective, build_target):
    workspace.configure(multiobjective=multiobjective, build_target=build_target)

    validate_targets()  # no targets.xlsx and no error


def test_experiments_stop_before_running_when_targets_are_missing(workspace):
    workspace.configure(experiments=[{"name": "only", "overrides": {}}])

    with pytest.raises(RuntimeError, match="ausente"):
        runner.run_experiments()
