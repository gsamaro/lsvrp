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
        (("instance", "dir"), "", "instance.dir"),
        (("instance", "files"), "DATA_PRP_20C", "instance.files"),
    ],
)
def test_normalize_rejects_invalid_runtime_controls(path, value, message):
    configured = copy.deepcopy(_valid_config())
    configured[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        Config.normalize(configured)
