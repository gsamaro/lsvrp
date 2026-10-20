import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from config.config import Config


def _valid_config():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_normalize_adds_relaxed_solution_use_default():
    configured = _valid_config()
    configured["relaxed_solution"].pop("use")

    normalized = Config.normalize(configured)

    assert normalized["relaxed_solution"]["use"] is False


def test_normalize_adds_target_use_relaxation_default():
    configured = _valid_config()
    configured["relaxed_solution"].pop("target_use_relaxation")

    normalized = Config.normalize(configured)

    assert normalized["relaxed_solution"]["target_use_relaxation"] is True


def test_normalize_adds_symmetry_breaking_default():
    configured = _valid_config()
    configured["solver"].pop("symmetry_breaking")

    normalized = Config.normalize(configured)

    assert normalized["solver"]["symmetry_breaking"]["hc1"] is False


def test_normalize_adds_coelho_inequalities_default():
    configured = _valid_config()
    configured["solver"].pop("coelho_inequalities")

    normalized = Config.normalize(configured)

    assert normalized["solver"]["coelho_inequalities"] is False


def test_normalize_adds_rounded_capacity_inequalities_defaults():
    configured = _valid_config()
    configured["solver"].pop("rounded_capacity_inequalities")

    normalized = Config.normalize(configured)

    assert normalized["solver"]["rounded_capacity_inequalities"] == {
        "enabled": False,
        "node_frequency": 50,
        "max_non_root_node": 200,
        "max_cuts_per_callback": 20,
        "max_cuts_per_non_root_callback": 3,
        "min_violation": 1e-6,
    }


def test_normalize_adds_positive_only_deviations_default():
    configured = _valid_config()
    configured["solver"].pop("goal_programming")

    normalized = Config.normalize(configured)

    assert normalized["solver"]["goal_programming"]["positive_only_deviations"] is True


def test_normalize_adds_objective_scale_default():
    configured = _valid_config()
    configured["solver"].pop("objective_scale", None)

    normalized = Config.normalize(configured)

    assert normalized["solver"]["objective_scale"] == 1.0


def test_normalize_keeps_objective_scale():
    configured = _valid_config()
    configured["solver"]["objective_scale"] = 10000

    normalized = Config.normalize(configured)

    assert normalized["solver"]["objective_scale"] == 10000.0


def test_normalize_adds_strengthened_bounds_default():
    configured = _valid_config()
    configured["solver"].pop("strengthened_bounds")

    normalized = Config.normalize(configured)

    assert normalized["solver"]["strengthened_bounds"] is True


def test_normalize_accepts_disabled_strengthened_bounds():
    configured = _valid_config()
    configured["solver"]["strengthened_bounds"] = False

    normalized = Config.normalize(configured)

    assert normalized["solver"]["strengthened_bounds"] is False


def test_normalize_accepts_current_config_file():
    normalized = Config.normalize(_valid_config())

    assert normalized["solver"]["timeLimit"] == 3600
    assert normalized["solver"]["method"] == "PSO"


def test_snapshot_returns_effective_config_as_an_isolated_copy():
    effective_config = Config.normalize(_valid_config())

    with patch.object(Config, "_data", effective_config):
        snapshot = Config.snapshot()
        snapshot["solver"]["symmetry_breaking"]["hc1"] = True

        next_snapshot = Config.snapshot()

    assert next_snapshot["relaxed_solution"]["use"] is False
    assert next_snapshot["solver"]["symmetry_breaking"]["hc1"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("solver", "timeLimit"), 0, "timeLimit"),
        (("solver", "method"), "invalid", "method"),
        (("solver", "threadsLimit"), 0, "threadsLimit"),
        (("workers", "num"), -1, "workers.num"),
        (("logging", "level"), "verbose", "logging.level"),
        (("solver", "multiobjective"), "true", "multiobjective"),
        (("solver", "objective_scale"), 0, "objective_scale"),
        (("solver", "objective_scale"), -1, "objective_scale"),
        (("solver", "objective_scale"), "1e4", "objective_scale"),
        (("solver", "objective_scale"), True, "objective_scale"),
        (("solver", "objective_scale"), float("inf"), "objective_scale"),
        (("solver", "strengthened_bounds"), "false", "strengthened_bounds"),
        (("solver", "symmetry_breaking", "hc1"), "false", "symmetry_breaking.hc1"),
        (("solver", "coelho_inequalities"), "false", "coelho_inequalities"),
        (
            ("solver", "rounded_capacity_inequalities", "enabled"),
            "false",
            "rounded_capacity_inequalities.enabled",
        ),
        (
            ("solver", "rounded_capacity_inequalities", "node_frequency"),
            0,
            "rounded_capacity_inequalities.node_frequency",
        ),
        (
            ("solver", "goal_programming", "positive_only_deviations"),
            "false",
            "goal_programming.positive_only_deviations",
        ),
        (
            ("relaxed_solution", "target_use_relaxation"),
            "false",
            "relaxed_solution.target_use_relaxation",
        ),
        (("instance", "dir"), "", "instance.dir"),
        (("instance", "files"), "DATA_PRP_20C", "instance.files"),
    ],
)
def test_normalize_rejects_invalid_runtime_controls(path, value, message):
    configured = copy.deepcopy(_valid_config())
    target = configured
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        Config.normalize(configured)


# ------------------------------------------------------------------ experiments


def _with_experiments(experiments, **solver):
    configured = _valid_config()
    configured.pop("experiments", None)
    configured["solver"].update(solver)
    configured["instance"]["output"] = "./out/"
    configured["experiments"] = experiments
    return Config.normalize(configured)


def test_experiments_default_to_a_single_run_with_the_base_output(monkeypatch):
    configured = _valid_config()
    configured.pop("experiments", None)
    monkeypatch.setattr(Config, "_data", Config.normalize(configured))

    resolved = Config.experiments()

    assert [name for name, _ in resolved] == ["default"]
    assert resolved[0][1]["instance"]["output"] == configured["instance"]["output"]
    assert "experiments" not in resolved[0][1]


def test_experiments_merge_overrides_in_depth(monkeypatch):
    base = _with_experiments(
        [
            {
                "name": "mat",
                "overrides": {
                    "solver": {
                        "matheuristic": {
                            "enabled": True,
                            "restricted": {"time_limit": 3600},
                        },
                        "polishing": {"after_seconds": 3000},
                    }
                },
            },
            {"name": "plain"},
        ]
    )
    monkeypatch.setattr(Config, "_data", base)

    (mat_name, mat), (plain_name, plain) = Config.experiments()

    assert (mat_name, plain_name) == ("mat", "plain")
    assert mat["instance"]["output"] == "./out/mat/"
    assert plain["instance"]["output"] == "./out/plain/"
    assert mat["solver"]["matheuristic"]["enabled"] is True
    assert mat["solver"]["matheuristic"]["restricted"]["time_limit"] == 3600
    # sibling keys of the override survive and the other experiment is untouched
    assert mat["solver"]["matheuristic"]["tsp"] == base["solver"]["matheuristic"]["tsp"]
    assert mat["solver"]["polishing"]["after_seconds"] == 3000
    assert plain["solver"]["polishing"]["after_seconds"] is None
    assert plain["solver"]["matheuristic"] == base["solver"]["matheuristic"]
    assert mat["experiment"] == "mat" and plain["experiment"] == "plain"
    assert mat["solver"]["telemetry"]["experiment_id"] == "mat"
    assert "experiments" not in mat


@pytest.mark.parametrize(
    ("experiments", "message"),
    [
        ([{"name": "a"}, {"name": "a"}], "repetido"),
        ([{"name": "a b"}], "name"),
        ([{"name": ""}], "name"),
        ([{"overrides": {}}], "name"),
        ({"name": "a"}, "lista"),
        ([{"name": "a", "overrides": "x"}], "overrides"),
    ],
)
def test_invalid_experiment_lists_are_rejected(experiments, message):
    with pytest.raises(ValueError, match=message):
        _with_experiments(experiments)


@pytest.mark.parametrize(
    "overrides",
    [
        {"instance": {"output": "./other/"}},
        {"workers": {"num": 2}},
        {"experiments": []},
    ],
)
def test_forbidden_overrides_are_rejected(monkeypatch, overrides):
    monkeypatch.setattr(
        Config, "_data", _with_experiments([{"name": "a", "overrides": overrides}])
    )

    with pytest.raises(ValueError, match="não pode alterar"):
        Config.experiments()


def test_an_invalid_override_fails_before_anything_runs(monkeypatch):
    monkeypatch.setattr(
        Config,
        "_data",
        _with_experiments(
            [
                {"name": "ok"},
                {
                    "name": "bad",
                    "overrides": {"solver": {"polishing": {"after_seconds": -1}}},
                },
            ]
        ),
    )

    with pytest.raises(ValueError, match="experiments\\[bad\\].*polishing"):
        Config.experiments()


def test_experiment_output_cannot_collide_with_the_targets_folder(monkeypatch):
    configured = _with_experiments([{"name": "target"}])
    configured["postprocessing"]["output"] = "./out/target/"
    monkeypatch.setattr(Config, "_data", configured)

    with pytest.raises(ValueError, match="coincide"):
        Config.experiments()


def test_replace_installs_a_normalized_copy(monkeypatch):
    monkeypatch.setattr(Config, "_data", None)
    snapshot = {"solver": {"timeLimit": 77}}

    Config.replace(snapshot)
    snapshot["solver"]["timeLimit"] = 1

    assert Config.get_nested("solver", "timeLimit") == 77
    assert Config.get_nested("solver", "objective_scale") == 1.0


def test_production_config_declares_three_experiments_sharing_one_base():
    configured = Config.normalize(_valid_config())

    names = [item["name"] for item in configured["experiments"]]
    assert names == ["pso_mip_polish", "matheuristic_mip_polish", "matheuristic"]
    assert configured["solver"]["total_time_budget"] is True
    assert configured["solver"]["threadsLimit"] == 1
    assert configured["solver"]["aggregate_vehicles"] is True
