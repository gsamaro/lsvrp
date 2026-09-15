import copy
import json
from pathlib import Path

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


def test_normalize_accepts_current_config_file():
    normalized = Config.normalize(_valid_config())

    assert normalized["solver"]["timeLimit"] == 3600
    assert normalized["solver"]["method"] == "PSO"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("solver", "timeLimit"), 0, "timeLimit"),
        (("solver", "method"), "invalid", "method"),
        (("solver", "threadsLimit"), 0, "threadsLimit"),
        (("workers", "num"), -1, "workers.num"),
        (("logging", "level"), "verbose", "logging.level"),
        (("solver", "multiobjective"), "true", "multiobjective"),
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
