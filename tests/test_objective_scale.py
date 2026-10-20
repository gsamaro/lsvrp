"""Global scale of the goal programming objective inside CPLEX.

The marginal cost of a decision variable in the multiobjective model is about
``(1 - alpha) * w * c / target``, below the CPLEX optimality tolerance, so an
unscaled LP can stop above its optimum. ``solver.objective_scale`` multiplies the
objective handed to CPLEX; every objective value and bound read back must be
divided by it, so the reported unit does not change.
"""

import importlib.util
from unittest.mock import Mock

import numpy as np
import pytest

from config import Config
from src.solvers._solver_common import ProblemData, evaluate_configured_objective
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)

HAS_CPLEX = importlib.util.find_spec("cplex") is not None


def _instance(customers=4, vehicles=2, periods=3, capacity=7, demand=3, stock=3):
    nodes = customers + 1
    coordinates = [(0, 0), (4, 0), (8, 0), (0, 4), (0, 8), (5, 5)][:nodes]
    return {
        "file": "unit.dat",
        "num_products": 1,
        "num_customers": customers,
        "num_periods": periods,
        "num_vehicles": vehicles,
        "B": 100,
        "b_p": [1],
        "c_p": [1],
        "s_p": [10],
        "M": 100,
        "U_pi": [[50] + [stock] * customers],
        "I_pi0": [[0] * nodes],
        "h_pi": [[0] + [1] * customers],
        "C": capacity,
        "f": 20,
        "a_ik": [
            [abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in coordinates]
            for a in coordinates
        ],
        "d_pit": [[[demand] * periods for _ in range(customers)]],
        "weight": [0.2] * 5,
        "alpha": 0.5,
        "targets": [
            {f"f{j}_target": 10.0 * j for j in range(1, 6)} for _ in range(periods)
        ],
    }


def _configure(monkeypatch, scale, multiobjective=True, relaxed=True):
    Config.load()
    value = Config.normalize(
        {
            "solver": {
                "multiobjective": multiobjective,
                "method": "GUROBY",
                "objective_scale": scale,
            },
            "relaxed_solution": {"use": relaxed},
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    return value


def _build(monkeypatch, scale, multiobjective=True):
    _configure(monkeypatch, scale, multiobjective)
    solver = MPPRP(_instance(), "/tmp", Mock(), {"start": False})
    solver.solver(timeLimit=1, numThreads=1)
    return solver


# ------------------------------------------------------------------------- config


def test_default_scale_is_neutral():
    assert Config.normalize({})["solver"]["objective_scale"] == 1.0


# ------------------------------------------------------------------ model assembly


def test_objective_handed_to_cplex_is_scaled(monkeypatch):
    plain = _build(monkeypatch, 1.0)
    scaled = _build(monkeypatch, 1e5)
    try:
        assert plain.objective_scale == 1.0
        assert scaled.objective_scale == 1e5
        plain_expr, scaled_expr = plain.model.objective_expr, scaled.model.objective_expr
        assert plain_expr.get_coef(plain.lambda_) == pytest.approx(0.5)
        assert scaled_expr.get_coef(scaled.lambda_) == pytest.approx(0.5 * 1e5)
        for key, variable in plain.positive.items():
            assert scaled_expr.get_coef(scaled.positive[key]) == pytest.approx(
                plain_expr.get_coef(variable) * 1e5
            )
        assert scaled.applied_features["objective_scale"] == 1e5
    finally:
        plain.terminate()
        scaled.terminate()


def test_single_objective_is_not_scaled(monkeypatch):
    solver = _build(monkeypatch, 1e5, multiobjective=False)
    try:
        assert solver.objective_scale == 1.0
    finally:
        solver.terminate()


def test_unscale_divides_values_and_keeps_missing(monkeypatch):
    solver = _build(monkeypatch, 1e4)
    try:
        assert solver._unscale(5e4) == pytest.approx(5.0)
        assert solver._unscale(None) is None
    finally:
        solver.terminate()


# ------------------------------------------------------------- CPLEX equivalence


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
def test_scaled_model_reports_the_same_objective_and_bound(monkeypatch):
    results = {}
    for scale in (1.0, 1e5):
        _configure(monkeypatch, scale, relaxed=False)
        solver = MPPRP(_instance(), "/tmp", Mock(), {"start": False})
        solver.solver(timeLimit=60, numThreads=2)
        try:
            assert "optimal" in str(solver.model.solve_details.status).lower()
            reported = solver.getResults()
            telemetry = solver.get_telemetry()
            variables = solver.get_variables()
            results[scale] = {
                "cplex": float(solver.model.objective_value),
                "fo": reported[7],
                "bound": reported[14],
                "telemetry_bound": telemetry["best_bound"],
                "telemetry_objective": telemetry["objective"],
                "gap": reported[8],
                "recomputed": evaluate_configured_objective(
                    ProblemData.from_map(_instance()), variables, Config.snapshot()
                ),
                "p": reported[6],
            }
        finally:
            solver.terminate()

    plain, scaled = results[1.0], results[1e5]
    # CPLEX sees the objective multiplied; the reported unit does not change.
    assert scaled["cplex"] == pytest.approx(plain["cplex"] * 1e5, rel=1e-6)
    for key in ("fo", "bound", "telemetry_bound", "telemetry_objective"):
        assert scaled[key] == pytest.approx(plain[key], rel=1e-6, abs=1e-9), key
    assert scaled["fo"] == pytest.approx(scaled["recomputed"], rel=1e-6, abs=1e-9)
    assert scaled["gap"] == pytest.approx(plain["gap"], abs=1e-9)
    assert np.asarray(scaled["p"]) == pytest.approx(np.asarray(plain["p"]), abs=1e-6)
