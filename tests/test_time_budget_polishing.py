"""Task time budget (solver.total_time_budget) and CPLEX solution polishing."""

from unittest.mock import Mock

import pytest

from config import Config
from src.helpers import TimeBudget
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)


@pytest.fixture(autouse=True)
def _clear_budget():
    TimeBudget.clear()
    yield
    TimeBudget.clear()


def _instance():
    coordinates = [(0, 0), (4, 0), (8, 0)]
    return {
        "file": "unit.dat",
        "num_products": 1,
        "num_customers": 2,
        "num_periods": 2,
        "num_vehicles": 1,
        "B": 100,
        "b_p": [1],
        "c_p": [1],
        "s_p": [10],
        "M": 100,
        "U_pi": [[50, 50, 50]],
        "I_pi0": [[0, 0, 0]],
        "h_pi": [[0, 1, 1]],
        "C": 10,
        "f": 20,
        "a_ik": [
            [abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in coordinates]
            for a in coordinates
        ],
        "d_pit": [[[2, 2], [2, 2]]],
        "weight": [0.2] * 5,
        "alpha": 0.5,
        "targets": [{f"f{j}_target": 10.0 * j for j in range(1, 6)}] * 2,
    }


def _build(monkeypatch, polishing=None, time_limit=None, **kwargs):
    Config.load()
    solver_cfg = {"multiobjective": True, "method": "GUROBY"}
    if polishing is not None:
        solver_cfg["polishing"] = {"after_seconds": polishing}
    monkeypatch.setattr(
        Config,
        "_data",
        Config.normalize(
            {"solver": solver_cfg, "relaxed_solution": {"use": True}}
        ),
    )
    solver = MPPRP(_instance(), "/tmp", Mock(), {"start": False}, **kwargs)
    solver.solver(timeLimit=time_limit, numThreads=1)
    return solver


def _polish_after(solver):
    return solver.model.context.cplex_parameters.mip.polishafter.time.get()


# ----------------------------------------------------------------------- config


def test_defaults_keep_the_previous_behaviour():
    normalized = Config.normalize({})["solver"]
    assert normalized["total_time_budget"] is False
    assert normalized["polishing"] == {"after_seconds": None}


@pytest.mark.parametrize("value", [0, -5, "300", True, float("inf")])
def test_polishing_after_seconds_is_validated(value):
    with pytest.raises(ValueError, match="polishing.after_seconds"):
        Config.normalize({"solver": {"polishing": {"after_seconds": value}}})


def test_total_time_budget_must_be_boolean():
    with pytest.raises(ValueError, match="total_time_budget"):
        Config.normalize({"solver": {"total_time_budget": "true"}})


# ------------------------------------------------------------------- time budget


def test_clamp_is_neutral_without_a_budget():
    assert TimeBudget.active() is False
    assert TimeBudget.clamp(120) == 120
    assert TimeBudget.remaining() is None
    assert TimeBudget.elapsed() is None


def test_clamp_limits_to_the_time_left_and_keeps_a_floor(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(TimeBudget.time, "perf_counter", lambda: clock["now"])
    TimeBudget.start(3600)
    clock["now"] = 100.0 + 1000.0

    assert TimeBudget.elapsed() == pytest.approx(1000.0)
    assert TimeBudget.clamp(3600) == pytest.approx(2600.0)
    assert TimeBudget.clamp(100) == 100  # already below what is left

    clock["now"] = 100.0 + 3599.9
    assert TimeBudget.clamp(3600) == 1.0  # never below one second

    TimeBudget.clear()
    assert TimeBudget.active() is False


def test_solver_receives_the_remaining_time(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(TimeBudget.time, "perf_counter", lambda: clock["now"])
    TimeBudget.start(3600)
    clock["now"] = 600.0

    solver = _build(monkeypatch, time_limit=3600)
    try:
        assert solver.model.context.cplex_parameters.timelimit.get() == pytest.approx(
            3000.0
        )
    finally:
        solver.terminate()


def test_solver_keeps_its_limit_without_a_budget(monkeypatch):
    solver = _build(monkeypatch, time_limit=3600)
    try:
        assert solver.model.context.cplex_parameters.timelimit.get() == 3600
    finally:
        solver.terminate()


# --------------------------------------------------------------------- polishing


def test_polishing_is_off_by_default(monkeypatch):
    solver = _build(monkeypatch)
    try:
        assert _polish_after(solver) > 1e20  # CPLEX default: never
        assert "polishing_after_seconds" not in solver.applied_features
    finally:
        solver.terminate()


def test_polishing_counts_from_the_mip_start_without_a_budget(monkeypatch):
    solver = _build(monkeypatch, polishing=3000)
    try:
        assert _polish_after(solver) == pytest.approx(3000.0)
        assert solver.applied_features["polishing_after_seconds"] == pytest.approx(
            3000.0
        )
    finally:
        solver.terminate()


def test_polishing_counts_from_the_task_start_with_a_budget(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(TimeBudget.time, "perf_counter", lambda: clock["now"])
    TimeBudget.start(3600)
    clock["now"] = 450.0  # PSO and model construction already spent 450 s

    solver = _build(monkeypatch, polishing=3000, time_limit=3600)
    try:
        assert _polish_after(solver) == pytest.approx(2550.0)
    finally:
        solver.terminate()


def test_polishing_starts_at_least_one_second_ahead(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(TimeBudget.time, "perf_counter", lambda: clock["now"])
    TimeBudget.start(3600)
    clock["now"] = 3200.0  # already past the requested instant

    solver = _build(monkeypatch, polishing=3000, time_limit=3600)
    try:
        assert _polish_after(solver) == pytest.approx(1.0)
    finally:
        solver.terminate()


@pytest.mark.parametrize(
    "restriction",
    [
        {"arc_order": [0, 1, 2, 0]},
        {"fixed_plan": {"X": [[0, 0]]}},
        {"routing_profile": {"mode": "ordered", "order": [0, 1, 2, 0]}},
    ],
)
def test_polishing_only_applies_to_the_unrestricted_mip(monkeypatch, restriction):
    solver = _build(monkeypatch, polishing=3000)
    try:
        assert _polish_after(solver) == pytest.approx(3000.0)  # unrestricted: applied
        solver.model.context.cplex_parameters.mip.polishafter.time.reset()
        solver.applied_features.pop("polishing_after_seconds")
        for name, value in restriction.items():
            setattr(solver, name, value)

        solver._apply_polishing()

        assert _polish_after(solver) > 1e20
        assert "polishing_after_seconds" not in solver.applied_features
    finally:
        solver.terminate()
