"""Target generation job: state, stale folders and resume (no CPLEX)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import src.process.WorkerProcess as worker_module
from config import Config
from src.helpers.FaceNadirTargets import TARGET_COLUMNS
from src.helpers.TargetsLoader import load_targets_by_file
from src.process import TargetGenerationProcess as generation

PERIODS = 3


def fake_compute(instance, weights, nadir_method="face", epsilon=1e-6,
                 lp_algorithm="dual", alpha=0.99):
    """LP stand-in: values depend only on the instance name and the settings."""
    if instance in FakeLP.fail_on:
        raise RuntimeError("falha simulada")
    FakeLP.calls.append(instance)
    seed = sum(map(ord, os.path.basename(instance)))
    rng = np.random.default_rng(seed)
    vertex = rng.uniform(1, 100, size=(5, 5, PERIODS))
    face = None
    if nadir_method == "face":
        face = vertex + rng.uniform(0, 50, size=vertex.shape)
        for j in range(5):
            face[j, j, :] = np.nan
    return {
        "z": vertex.sum(axis=2).diagonal(),
        "vertex": vertex,
        "face_minima": face,
        "periods": PERIODS,
        "features": {"symmetric_routes": True},
        "seconds": 0.1,
        "build_seconds": 0.05,
    }


class FakeLP:
    calls = []
    fail_on = set()


@pytest.fixture
def job(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for folder, names in {"DATA_PRP_5C": ("PRP1_C5.dat", "PRP2_C5.dat"),
                          "DATA_PRP_10C": ("PRP3_C10.dat",)}.items():
        (data / folder).mkdir(parents=True)
        for name in names:
            (data / folder / name).write_text("x")
    (data / "DATA_PRP_5C" / "PRP31_C5.dat").write_text("x")  # fora do escopo PRP 1-30

    monkeypatch.setattr(generation, "compute_instance_lps", fake_compute)
    monkeypatch.setattr(worker_module, "MPI_BOOL", False)
    monkeypatch.setattr(
        generation, "git_state", lambda: {"commit": "c" * 40, "dirty": False}
    )
    FakeLP.calls = []
    FakeLP.fail_on = set()

    saved = Config.snapshot()
    snapshot = Config.snapshot()
    snapshot["instance"]["dir"] = f"{data}/"
    snapshot["workers"]["num"] = 1
    snapshot["target_generation"]["instance"]["files"] = [
        "DATA_PRP_5C",
        "DATA_PRP_10C",
    ]
    snapshot["target_generation"]["output"] = str(tmp_path / "out" / "target")
    Config.replace(snapshot)
    yield tmp_path / "out" / "target"
    Config.replace(saved)


def _change(path, value):
    snapshot = Config.snapshot()
    node = snapshot
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    Config.replace(snapshot)


def test_run_writes_the_targets_file_without_provenance_columns_and_the_state(job):
    state = generation.run_target_generation(Config)

    assert state["status"] == "complete"
    assert state["found_tasks"] == state["expected_tasks"] == 3  # PRP31 fica de fora
    frame = pd.read_excel(job / "targets.xlsx", engine="openpyxl")
    assert len(frame) == 3 * PERIODS
    assert list(frame.columns[:2]) == ["file", "time"]
    assert list(frame.columns) == ["file", "time"] + TARGET_COLUMNS
    saved = json.loads((job / "targets_state.json").read_text())
    assert saved["fingerprint"] == state["fingerprint"]
    assert saved["commit"] == "c" * 40
    assert saved["instances"] == sorted(saved["instances"])
    assert saved["targets_sha256"] == generation.file_sha256(job / "targets.xlsx")
    assert saved["config"]["solver"]["symmetric_routes"] is True
    assert saved["config"]["postprocessing"]["build_target"] is True
    assert saved["rows"] == len(frame)


def test_targets_keep_the_columns_the_loader_and_union_results_use(job):
    generation.run_target_generation(Config)

    frame = pd.read_excel(job / "targets.xlsx", engine="openpyxl")
    for j in range(1, 6):
        assert f"f{j}_target" in frame.columns
    assert {"file", "time"} <= set(frame.columns)
    by_file = load_targets_by_file(str(job / "targets.xlsx"))
    assert len(by_file) == 3
    assert all(len(rows) == PERIODS for rows in by_file.values())
    ideal = frame[[f"f{j}_ideal" for j in range(1, 6)]].to_numpy()
    nadir = frame[[f"f{j}_nadir" for j in range(1, 6)]].to_numpy()
    target = frame[[f"f{j}_target" for j in range(1, 6)]].to_numpy()
    assert np.allclose(target, ideal + 0.3 * (nadir - ideal))


def test_same_configuration_is_skipped_without_solving_again(job):
    generation.run_target_generation(Config)
    calls = len(FakeLP.calls)

    state = generation.run_target_generation(Config)

    assert state["status"] == "complete"
    assert len(FakeLP.calls) == calls
    assert not list(job.parent.glob("target.stale-*"))


def test_changed_configuration_moves_the_previous_folder_aside(job):
    generation.run_target_generation(Config)
    old_hash = generation.file_sha256(job / "targets.xlsx")
    _change(("target_generation", "factor"), 0.5)

    state = generation.run_target_generation(Config)

    stale = list(job.parent.glob("target.stale-*"))
    assert len(stale) == 1
    assert generation.file_sha256(stale[0] / "targets.xlsx") == old_hash
    assert (stale[0] / "work").is_dir()
    assert state["status"] == "complete"
    frame = pd.read_excel(job / "targets.xlsx", engine="openpyxl")
    ideal = frame["f1_ideal"].to_numpy()
    assert np.allclose(
        frame["f1_target"].to_numpy(),
        ideal + 0.5 * (frame["f1_nadir"].to_numpy() - ideal),
    )
    assert len(FakeLP.calls) == 6  # tudo recalculado, nada misturado


def test_legacy_targets_without_state_are_moved_aside_not_deleted(job):
    job.mkdir(parents=True)
    legacy = job / "targets.xlsx"
    pd.DataFrame({"file": ["x"], "time": [0]}).to_excel(legacy, index=False)
    legacy_hash = generation.file_sha256(legacy)

    state = generation.run_target_generation(Config)

    stale = list(job.parent.glob("target.stale-*"))
    assert len(stale) == 1
    assert generation.file_sha256(stale[0] / "targets.xlsx") == legacy_hash
    assert state["status"] == "complete"
    assert generation.read_state(str(job))["status"] == "complete"


def test_interrupted_run_resumes_from_the_computed_instances(job):
    failing = "PRP2_C5.dat"
    FakeLP.fail_on = {f"{Config.get_nested('instance', 'dir')}DATA_PRP_5C/{failing}"}

    state = generation.run_target_generation(Config)

    assert state["status"] == "incomplete"
    assert state["found_tasks"] == 2
    assert not (job / "targets.xlsx").exists()
    assert generation.read_state(str(job))["status"] == "incomplete"

    FakeLP.fail_on = set()
    FakeLP.calls = []
    state = generation.run_target_generation(Config)

    assert state["status"] == "complete"
    assert [os.path.basename(call) for call in FakeLP.calls] == [failing]
    assert not list(job.parent.glob("target.stale-*"))


def test_edited_targets_file_is_rebuilt_from_the_cached_results(job):
    generation.run_target_generation(Config)
    pd.DataFrame({"file": ["x"]}).to_excel(job / "targets.xlsx", index=False)
    FakeLP.calls = []

    state = generation.run_target_generation(Config)

    assert FakeLP.calls == []
    assert state["status"] == "complete" and state["rows"] == 3 * PERIODS
    assert state["targets_sha256"] == generation.file_sha256(job / "targets.xlsx")
    assert len(pd.read_excel(job / "targets.xlsx")) == 3 * PERIODS


def test_runs_max_nadir_does_not_solve_face_lps_and_matches_the_old_rule(job):
    _change(("target_generation", "nadir", "method"), "runs_max")

    generation.run_target_generation(Config)

    frame = pd.read_excel(job / "targets.xlsx", engine="openpyxl")
    first = frame[frame["file"] == frame["file"].iloc[0]]
    vertex = fake_compute(first["file"].iloc[0], None, "runs_max")["vertex"]
    assert np.allclose(
        first[[f"f{j}_nadir" for j in range(1, 6)]].to_numpy().T, vertex.max(axis=0)
    )
    assert np.allclose(
        first[[f"f{j}_ideal" for j in range(1, 6)]].to_numpy().T, vertex.min(axis=0)
    )


def test_fingerprint_ignores_workers_and_logging_but_not_the_method(job):
    def fingerprint():
        settings = generation.generation_settings(Config)
        return generation.experiment_fingerprint(generation.effective_config(settings))

    base = fingerprint()
    _change(("workers", "num"), 7)
    _change(("logging", "level"), "DEBUG")
    assert fingerprint() == base

    _change(("target_generation", "lp", "algorithm"), "primal")
    assert fingerprint() != base


def test_plan_reports_the_action_without_creating_anything(job):
    plan = generation.plan_generation(Config)

    assert plan["action"] == "run" and plan["instances"] == 3
    assert not job.exists()
    assert FakeLP.calls == []


def test_a_task_failure_is_reported_instead_of_raised(job, monkeypatch):
    FakeLP.fail_on = {"./data/x.dat"}
    result = generation.solve_instance_task(
        {
            "instance": "./data/x.dat",
            "fingerprint": "f",
            "config_json": json.dumps(Config.snapshot()),
            "weights": [],
            "nadir_method": "face",
            "epsilon": 1e-6,
            "lp_algorithm": "dual",
            "cache_path": str(job / "work" / "x.json"),
        }
    )

    assert "falha simulada" in result["error"]
