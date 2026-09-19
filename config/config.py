import copy
import json
from pathlib import Path


class Config:
    _data = None
    _DEFAULTS = {
        "solver": {
            "threadsLimit": 1,
            "timeLimit": 3600,
            "method": "PSO",
            "multiobjective": True,
            "strengthened_bounds": True,
            "goal_programming": {
                "positive_only_deviations": True,
            },
            "symmetry_breaking": {"hc1": False},
            "coelho_inequalities": False,
            "rounded_capacity_inequalities": {
                "enabled": False,
                "node_frequency": 50,
                "max_non_root_node": 200,
                "max_cuts_per_callback": 20,
                "max_cuts_per_non_root_callback": 3,
                "min_violation": 1e-6,
            },
        },
        "workers": {"num": "auto"},
        "logging": {"level": "INFO"},
        "instance": {"dir": "./data/", "output": "./out/", "files": []},
        "relaxed_solution": {
            "replace_model": True,
            "use": False,
            "target_use_relaxation": True,
        },
        "postprocessing": {"build_target": False, "output": "./out/target/"},
    }

    @classmethod
    def load(cls):
        if cls._data is None:
            config_path = Path(__file__).parent / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                cls._data = cls.normalize(json.load(f))

    @classmethod
    def _merge_defaults(cls, defaults, configured):
        merged = {}
        configured = configured or {}
        for key, default in defaults.items():
            value = configured.get(key, default)
            if isinstance(default, dict):
                if value is not None and not isinstance(value, dict):
                    raise ValueError(f"config.{key} deve ser um objeto")
                merged[key] = cls._merge_defaults(default, value)
            else:
                merged[key] = value

        for key, value in configured.items():
            if key not in merged:
                merged[key] = value
        return merged

    @classmethod
    def normalize(cls, configured):
        if not isinstance(configured, dict):
            raise ValueError("config deve ser um objeto JSON")

        normalized = cls._merge_defaults(cls._DEFAULTS, configured)
        cls.validate(normalized)
        return normalized

    @classmethod
    def validate(cls, configured):
        solver = configured["solver"]
        time_limit = solver["timeLimit"]
        try:
            time_limit = int(time_limit)
        except (TypeError, ValueError) as error:
            raise ValueError("config.solver.timeLimit deve ser um inteiro positivo") from error
        if isinstance(solver["timeLimit"], bool) or time_limit <= 0:
            raise ValueError("config.solver.timeLimit deve ser um inteiro positivo")
        solver["timeLimit"] = time_limit

        method = str(solver["method"]).upper()
        if method not in {"GUROBY", "PSO"}:
            raise ValueError("config.solver.method deve ser GUROBY ou PSO")
        solver["method"] = method

        threads_limit = solver["threadsLimit"]
        if threads_limit is None or threads_limit == "None":
            solver["threadsLimit"] = None
        elif isinstance(threads_limit, bool) or not isinstance(threads_limit, int) or threads_limit <= 0:
            raise ValueError("config.solver.threadsLimit deve ser um inteiro positivo ou None")

        cls._validate_bool(solver["multiobjective"], "config.solver.multiobjective")
        cls._validate_bool(
            solver["strengthened_bounds"], "config.solver.strengthened_bounds"
        )
        goal_programming = solver["goal_programming"]
        cls._validate_bool(
            goal_programming["positive_only_deviations"],
            "config.solver.goal_programming.positive_only_deviations",
        )
        cls._validate_bool(
            solver["coelho_inequalities"], "config.solver.coelho_inequalities"
        )
        rounded_capacity = solver["rounded_capacity_inequalities"]
        cls._validate_bool(
            rounded_capacity["enabled"],
            "config.solver.rounded_capacity_inequalities.enabled",
        )
        for key in (
            "node_frequency",
            "max_non_root_node",
            "max_cuts_per_callback",
            "max_cuts_per_non_root_callback",
        ):
            value = rounded_capacity[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"config.solver.rounded_capacity_inequalities.{key} "
                    "deve ser um inteiro positivo"
                )
        min_violation = rounded_capacity["min_violation"]
        if isinstance(min_violation, bool) or not isinstance(min_violation, (int, float)) or min_violation < 0:
            raise ValueError(
                "config.solver.rounded_capacity_inequalities.min_violation "
                "deve ser um numero nao negativo"
            )
        symmetry_breaking = solver["symmetry_breaking"]
        cls._validate_bool(symmetry_breaking["hc1"], "config.solver.symmetry_breaking.hc1")
        cls._validate_workers(configured["workers"]["num"])

        level = configured["logging"]["level"]
        if not isinstance(level, str) or level.upper() not in {"DEBUG", "INFO", "OFF"}:
            raise ValueError("config.logging.level deve ser DEBUG, INFO ou OFF")
        configured["logging"]["level"] = level.upper()

        instance = configured["instance"]
        cls._validate_path(instance["dir"], "config.instance.dir")
        cls._validate_path(instance["output"], "config.instance.output")
        if not isinstance(instance["files"], list) or not all(
            isinstance(value, str) and value.strip() for value in instance["files"]
        ):
            raise ValueError("config.instance.files deve ser uma lista de strings não vazias")

        relaxed = configured["relaxed_solution"]
        cls._validate_bool(relaxed["replace_model"], "config.relaxed_solution.replace_model")
        cls._validate_bool(relaxed["use"], "config.relaxed_solution.use")
        cls._validate_bool(
            relaxed["target_use_relaxation"],
            "config.relaxed_solution.target_use_relaxation",
        )

        postprocessing = configured["postprocessing"]
        cls._validate_bool(postprocessing["build_target"], "config.postprocessing.build_target")
        cls._validate_path(postprocessing["output"], "config.postprocessing.output")

    @staticmethod
    def _validate_bool(value, path):
        if not isinstance(value, bool):
            raise ValueError(f"{path} deve ser booleano")

    @staticmethod
    def _validate_path(value, path):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} deve ser uma string não vazia")

    @staticmethod
    def _validate_workers(value):
        if isinstance(value, bool):
            raise ValueError("config.workers.num deve ser auto, max, all, 0 ou inteiro positivo")
        if value is None or value == 0 or value == "0":
            return
        if isinstance(value, str) and value.strip().lower() in {"auto", "max", "all"}:
            return
        if not isinstance(value, int) or value <= 0:
            raise ValueError("config.workers.num deve ser auto, max, all, 0 ou inteiro positivo")

    @classmethod
    def get(cls, key, default=None):
        cls.load()
        return cls._data.get(key, default)

    @classmethod
    def snapshot(cls):
        """Return an isolated copy of the effective, normalized configuration."""
        cls.load()
        return copy.deepcopy(cls._data)

    @classmethod
    def get_nested(cls, *keys, default=None):
        cls.load()
        data = cls._data
        for key in keys:
            if not isinstance(data, dict):
                return default
            data = data.get(key)
        return data if data is not None else default
