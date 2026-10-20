"""Several experiments in one job: sequential runs, resume and pairing keys."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import Config
from src.process import ExperimentRunner as runner
from src.process.PostProcessingProcess import PostProcessingProcess

TASKS_PER_FILE = 2


class FakeWorker:
    """Stands in for WorkerProcess: counts tasks and records what was run."""

    runs = []
    fail_on = set()

    def __init__(self, num_workers=None, log=None, **kwargs):
        self.kwargs = kwargs

    def count_tasks(self, files):
        return len(files) * TASKS_PER_FILE

    def create_executor(self, total_tasks):
        return None

    def run_parallel(self, instancies, solver, executor=None):
        name = Config.get_nested("experiment")
        if name in FakeWorker.fail_on:
            raise RuntimeError(f"falha simulada em {name}")
        FakeWorker.runs.append(
            (name, Config.get_nested("instance", "output"), len(instancies), solver)
        )


@pytest.fixture
def job(tmp_path, monkeypatch):
    data = tmp_path / "data" / "DATA_PRP_5C"
    data.mkdir(parents=True)
    for number in (1, 2):
        (data / f"PRP{number}_C5.dat").write_text("x")
    FakeWorker.runs = []
    FakeWorker.fail_on = set()
    state = {"missing_tasks": 0}

    def fake_finalize(postprocessing, run_tag, build_target, experiment=None, fingerprint=None):
        files = len(list((data).iterdir()))
        keys = [f"inst{n}|w{w}|0.99" for n in range(files) for w in range(TASKS_PER_FILE)]
        keys = keys[: len(keys) - state["missing_tasks"]]
        frame = pd.DataFrame(
            {
                "task_key": keys,
                "experiment": experiment,
                "experiment_fingerprint": fingerprint,
            }
        )
        path = os.path.join(Config.get_nested("instance", "output"), f"{run_tag}-union_results.xlsx")
        frame.to_excel(path, index=False)
        return path

    def configure(overrides_by_name):
        experiments = [{"name": n, "overrides": o} for n, o in overrides_by_name.items()]
        monkeypatch.setattr(
            Config,
            "_data",
            Config.normalize(
                {
                    "instance": {
                        "dir": f"{tmp_path}/data/",
                        "output": f"{tmp_path}/out/",
                        "files": ["DATA_PRP_5C"],
                    },
                    "solver": {"method": "PSO", "multiobjective": False},
                    "postprocessing": {"output": f"{tmp_path}/targets/"},
                    "experiments": experiments,
                }
            ),
        )

    with patch.object(runner, "WorkerProcess", FakeWorker), patch.object(
        runner, "create_run_logger", return_value=MagicMock()
    ), patch.object(runner, "PostProcessingProcess", MagicMock()), patch.object(
        runner, "finalize_postprocessing", side_effect=fake_finalize
    ):
        yield type("Job", (), {"configure": staticmethod(configure), "state": state, "root": tmp_path})


TWO = {
    "first": {"solver": {"polishing": {"after_seconds": 3000}}},
    "second": {"solver": {"timeLimit": 99}},
}


def _state(job, name):
    return runner.read_state(f"{job.root}/out/{name}/")


def test_experiments_run_in_order_each_with_its_own_folder_and_state(job):
    job.configure(TWO)

    failed = runner.run_experiments()

    assert failed == []
    assert [run[0] for run in FakeWorker.runs] == ["first", "second"]
    assert [run[1] for run in FakeWorker.runs] == [
        f"{job.root}/out/first/",
        f"{job.root}/out/second/",
    ]
    for name in ("first", "second"):
        state = _state(job, name)
        assert state["status"] == "complete"
        assert state["found_tasks"] == state["expected_tasks"] == 2 * TASKS_PER_FILE
        assert os.path.isfile(state["union_path"])
        assert os.path.dirname(state["union_path"]) == os.path.abspath(
            f"{job.root}/out/{name}"
        )
    assert _state(job, "first")["fingerprint"] != _state(job, "second")["fingerprint"]


def test_a_second_submission_skips_completed_experiments(job):
    job.configure(TWO)
    runner.run_experiments()
    FakeWorker.runs = []

    assert runner.run_experiments() == []

    assert FakeWorker.runs == []


def test_resume_reads_the_union_of_the_right_experiment(job):
    job.configure(TWO)
    runner.run_experiments()
    FakeWorker.runs = []
    # The "first" state points to a union file that belongs to another experiment.
    first = _state(job, "first")
    second = _state(job, "second")
    first["union_path"] = second["union_path"]
    runner.write_state(f"{job.root}/out/first/", first)

    runner.run_experiments()

    assert [run[0] for run in FakeWorker.runs] == ["first"]  # only the wrong one reruns


def test_resume_never_trusts_a_glob_of_the_folder(job):
    job.configure(TWO)
    runner.run_experiments()
    FakeWorker.runs = []
    first = _state(job, "first")
    os.remove(first["union_path"])
    # A stray file that looks like a union result must not satisfy the check.
    stray = f"{job.root}/out/first/2000-01-01-aaaaaa-bbbbbb-union_results.xlsx"
    pd.DataFrame({"task_key": ["x"], "experiment": ["first"]}).to_excel(stray, index=False)

    runner.run_experiments()

    assert [run[0] for run in FakeWorker.runs] == ["first"]


def test_incomplete_union_is_reported_and_not_skipped_next_time(job):
    job.configure({"only": {}})
    job.state["missing_tasks"] = 1

    assert runner.run_experiments() == ["only"]
    assert _state(job, "only")["status"] == "incomplete"

    job.state["missing_tasks"] = 0
    FakeWorker.runs = []
    assert runner.run_experiments() == []
    assert [run[0] for run in FakeWorker.runs] == ["only"]
    assert _state(job, "only")["status"] == "complete"


def test_changed_configuration_moves_the_folder_aside_without_deleting(job):
    job.configure({"only": {"solver": {"timeLimit": 10}}})
    runner.run_experiments()
    old_union = _state(job, "only")["union_path"]
    FakeWorker.runs = []

    job.configure({"only": {"solver": {"timeLimit": 20}}})
    runner.run_experiments()

    stale = [p for p in os.listdir(f"{job.root}/out") if p.startswith("only.stale-")]
    assert len(stale) == 1
    assert os.path.isfile(os.path.join(f"{job.root}/out", stale[0], os.path.basename(old_union)))
    assert [run[0] for run in FakeWorker.runs] == ["only"]
    assert _state(job, "only")["status"] == "complete"
    assert not os.path.exists(old_union)  # the new folder starts clean


def test_results_without_state_are_not_mixed_with_the_new_run(job):
    job.configure({"only": {}})
    folder = f"{job.root}/out/only/"
    os.makedirs(folder)
    open(os.path.join(folder, "old_fobs.xlsx"), "w").write("old")

    runner.run_experiments()

    assert any(p.startswith("only.stale-") for p in os.listdir(f"{job.root}/out"))
    assert not os.path.exists(os.path.join(folder, "old_fobs.xlsx"))


def test_a_failing_experiment_does_not_stop_the_next_ones(job):
    job.configure(TWO)
    FakeWorker.fail_on = {"first"}

    failed = runner.run_experiments()

    assert failed == ["first"]
    assert [run[0] for run in FakeWorker.runs] == ["second"]
    assert _state(job, "first")["status"] == "running"  # never finished
    assert _state(job, "second")["status"] == "complete"


def test_invalid_experiment_aborts_before_any_run(job, monkeypatch):
    job.configure({"ok": {}, "bad": {"solver": {"polishing": {"after_seconds": -1}}}})

    with pytest.raises(ValueError, match="bad"):
        runner.run_experiments()

    assert FakeWorker.runs == []


def test_fingerprint_ignores_workers_and_logging_but_not_the_solver(job):
    job.configure({"only": {}})
    _, config = Config.experiments()[0]
    changed_runtime = dict(config, workers={"num": 7}, logging={"level": "DEBUG"})
    changed_solver = {**config, "solver": {**config["solver"], "timeLimit": 1}}

    assert runner.experiment_fingerprint(changed_runtime) == runner.experiment_fingerprint(config)
    assert runner.experiment_fingerprint(changed_solver) != runner.experiment_fingerprint(config)


# ------------------------------------------------------------------ pairing keys


def _results(file="./data/DATA_PRP_20C/PRP1_C20_P8_V5_T12_S1.dat"):
    return pd.DataFrame(
        {
            "file": [file, file],
            "weight": ["[0.2, 0.2, 0.2, 0.2, 0.2]"] * 2,
            "alpha": [0.99, 0.99],
            "time": [0, 1],
        }
    )


def test_pairing_keys_match_across_experiments_and_are_unique_within_one():
    first = PostProcessingProcess.add_pairing_keys(
        _results(), experiment="pso", fingerprint="f1", run_tag="r1"
    )
    second = PostProcessingProcess.add_pairing_keys(
        _results(), experiment="mat", fingerprint="f2", run_tag="r2"
    )

    assert list(first["row_key"]) == list(second["row_key"])
    assert list(first["task_key"]) == list(second["task_key"])
    assert first["row_key"].is_unique and first["task_key"].nunique() == 1
    assert first["row_key"].iloc[0].endswith("|0.99|0")
    assert first["row_key"].iloc[1].endswith("|0.99|1")
    assert set(first["experiment"]) == {"pso"} and set(second["experiment"]) == {"mat"}
    assert set(first["experiment_fingerprint"]) == {"f1"}
    assert set(first["experiment_run_tag"]) == {"r1"}


def test_pairing_keys_differ_by_instance_weight_and_alpha():
    base = _results()
    other_instance = _results("./data/DATA_PRP_20C/PRP2_C20_P8_V5_T12_S1.dat")
    other_weight = base.assign(weight="[0.35, 0.35, 0.1, 0.1, 0.1]")
    other_alpha = base.assign(alpha=0.01)
    keys = {
        PostProcessingProcess.add_pairing_keys(frame)["task_key"].iloc[0]
        for frame in (base, other_instance, other_weight, other_alpha)
    }

    assert len(keys) == 4


def test_pairing_keys_default_to_the_default_experiment():
    keyed = PostProcessingProcess.add_pairing_keys(_results())

    assert set(keyed["experiment"]) == {"default"}
    assert keyed["instance_key"].str.contains("PRP1_C20").all()
