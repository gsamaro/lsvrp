import copy
import importlib.util

import pytest

from config import Config
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cplex") is None,
    reason="CPLEX não está instalado neste ambiente",
)


class DummyLogger:
    def debug(self, message):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def _data():
    return {
        "num_products": 1,
        "num_customers": 1,
        "num_periods": 1,
        "num_vehicles": 1,
        "B": 10,
        "b_p": [1],
        "c_p": [2],
        "s_p": [3],
        "M": 100,
        "U_pi": [[20, 20]],
        "I_pi0": [[0, 0]],
        "h_pi": [[0, 1]],
        "C": 10,
        "f": 7,
        "a_ik": [[0, 11], [11, 0]],
        "d_pit": [[[5]]],
        "weight": [0.2] * 5,
        "alpha": 0.01,
        "targets": [],
    }


def _solve_with_rounding(monkeypatch, enabled):
    Config.load()
    configured = copy.deepcopy(Config._data)
    configured["solver"]["multiobjective"] = False
    configured["solver"]["rounded_capacity_inequalities"].update(
        {
            "enabled": enabled,
            "node_frequency": 1,
            "max_non_root_node": 2,
        }
    )
    configured["relaxed_solution"]["use"] = False
    configured["postprocessing"]["build_target"] = False
    monkeypatch.setattr(Config, "_data", configured)

    solver = MultProductProdctionRoutingProblem(
        map=_data(),
        dir="/tmp",
        log=DummyLogger(),
        start={"start": False},
    )
    try:
        solver.solver(timeLimit=3)
        return str(solver.model.solve_details.status), float(solver.model.objective_value)
    finally:
        solver.terminate()


def test_small_cplex_instance_solves_with_and_without_rounded_capacity_cuts(monkeypatch):
    status_without, objective_without = _solve_with_rounding(monkeypatch, False)
    status_with, objective_with = _solve_with_rounding(monkeypatch, True)

    assert "optimal" in status_without.lower()
    assert "optimal" in status_with.lower()
    assert objective_with == pytest.approx(objective_without)
