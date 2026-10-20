"""Matheuristic without the vehicle index, decoder-based start and clean plans.

Unit tests build models and starts; the CPLEX integration tests solve small
instances and compare the vehicle-free stages with the vehicle-indexed ones.
"""

import importlib.util
from unittest.mock import Mock

import numpy as np
import pytest

from config import Config
from src.solvers._routing_common import ordered_arcs, validate_variables
from src.solvers._solver_common import ProblemData
from src.solvers.DeterministicMatheuristic import DeterministicMatheuristic
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)

HAS_CPLEX = importlib.util.find_spec("cplex") is not None


def _data(customers=5, products=2, vehicles=2, periods=3, capacity=40, seed=0):
    rng = np.random.default_rng(seed)
    nodes = customers + 1
    xy = rng.integers(0, 30, size=(nodes, 2))
    distance = np.rint(np.linalg.norm(xy[:, None] - xy[None], axis=2)).astype(int)
    return {
        "file": "unit.dat",
        "num_products": products,
        "num_customers": customers,
        "num_periods": periods,
        "num_vehicles": vehicles,
        "B": 400,
        "b_p": [1, 2][:products],
        "c_p": [2, 3][:products],
        "s_p": [30, 50][:products],
        "M": 1000,
        "U_pi": [[200] + [60] * customers for _ in range(products)],
        "I_pi0": [[0] + [2] * customers for _ in range(products)],
        "h_pi": [[1] + [2] * customers for _ in range(products)],
        "C": capacity,
        "f": 25,
        "a_ik": distance.tolist(),
        "d_pit": rng.integers(1, 6, size=(products, customers, periods)).tolist(),
        "weight": [0.2] * 5,
        "alpha": 0.5,
        "targets": [
            {f"f{j}_target": 20.0 * j for j in range(1, 6)} for _ in range(periods)
        ],
    }


def _config(monkeypatch, fleet=False, start="decoder", multiobjective=False, **extra):
    value = Config.normalize(
        {
            "solver": {
                "multiobjective": multiobjective,
                "method": "GUROBY",
                "aggregate_vehicles": fleet,
                "matheuristic": {
                    "enabled": True,
                    "start": start,
                    "tsp": {"time_limit": 5},
                    "restricted": {"time_limit": 30},
                    "route_improvement": {"enabled": True, "time_limit": 15},
                },
                "pso": {
                    "swarm_size": 30,
                    "max_iterations": 30,
                    "execution_backend": "python",
                    "jit_warmup": False,
                    "lot_sizing_bounds": {"enabled": False},
                },
                **extra,
            },
            "relaxed_solution": {"use": False},
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    return value


def _build_only(configured):
    """Make ``solver()`` assemble the model without solving it."""
    configured["relaxed_solution"]["use"] = True


def _matheuristic(data):
    wrapper = DeterministicMatheuristic(data, "/tmp", Mock())
    wrapper.order = list(range(data["num_customers"] + 1)) + [0]
    return wrapper


# ------------------------------------------------------------------------ config


def test_config_defaults_and_validation_for_the_start():
    assert Config.normalize({})["solver"]["matheuristic"]["start"] == "decoder"
    for value in ("legacy", "decoder", "pso"):
        configured = Config.normalize({"solver": {"matheuristic": {"start": value}}})
        assert configured["solver"]["matheuristic"]["start"] == value
    with pytest.raises(ValueError, match="matheuristic.start"):
        Config.normalize({"solver": {"matheuristic": {"start": "random"}}})


# ------------------------------------------------------------------- start modes


@pytest.mark.parametrize("start", ["legacy", "decoder", "pso"])
def test_every_start_mode_returns_a_valid_ordered_solution(monkeypatch, start):
    _config(monkeypatch, start=start)
    data = _data()
    wrapper = _matheuristic(data)
    values = wrapper._constructive_start()
    assert values is not None
    problem = ProblemData.from_map(data)
    assert validate_variables(problem, values, ordered_arcs(wrapper.order))["feasible"]


def test_decoder_start_is_better_than_the_legacy_start(monkeypatch):
    _config(monkeypatch, start="legacy")
    data = _data(customers=6, periods=4)
    legacy = _matheuristic(data)
    legacy_score = legacy._score(legacy._constructive_start())[0]
    _config(monkeypatch, start="decoder")
    decoder = _matheuristic(data)
    assert decoder._score(decoder._constructive_start())[0] < legacy_score


def test_pso_start_is_not_worse_than_the_decoder_start(monkeypatch):
    _config(monkeypatch, start="decoder")
    data = _data(customers=6, periods=4)
    decoder = _matheuristic(data)
    decoder_score = decoder._score(decoder._constructive_start())[0]
    _config(monkeypatch, start="pso")
    pso = _matheuristic(data)
    assert pso._score(pso._constructive_start())[0] <= decoder_score + 1e-6


def test_pso_start_falls_back_to_the_decoder_without_a_pso_solution(monkeypatch):
    _config(monkeypatch, start="pso")
    wrapper = _matheuristic(_data())
    wrapper._pso_start = Mock(return_value=None)
    values = wrapper._constructive_start()
    assert values is not None
    wrapper._pso_start.assert_called_once()


# ---------------------------------------------------------- numerical residue


def test_delivery_totals_drop_numerical_residue():
    q = np.zeros((1, 2, 3, 1))
    q[0, 0, 1, 0] = 4.5
    q[0, 1, 1, 0] = 0.5
    q[0, 0, 2, 0] = 4e-12
    q[0, 1, 2, 0] = 3e-7
    totals = DeterministicMatheuristic._delivery_totals({"Q": q})
    assert totals[0, 1, 0] == pytest.approx(5.0)
    assert totals[0, 2, 0] == 0.0


# --------------------------------------------------------------- model structure


def test_arc_order_restricts_arcs_and_keeps_binary_variables(monkeypatch):
    _config(monkeypatch, fleet=True)
    data = _data(customers=4, vehicles=3)
    data["aggregate_vehicles"] = True
    model = MPPRP(data, "/tmp", Mock(), {"start": False}, arc_order=[0, 1, 2, 3, 4, 0])
    try:
        model.createDecisionVariables()
        # plant -> i, i -> j for i before j, i -> plant
        assert len(model.model.Z_v_i_k_t) == (4 + 6 + 4) * data["num_periods"]
        assert all(v.is_binary() for v in model.model.Z_v_i_k_t.values())
        assert model.model.Z_v_i_k_t[0, 2, 1, 0] == 0
        assert model.eta == {}
    finally:
        model.terminate()


def test_fixed_plan_without_eta_requires_visits_only_for_loaded_customers(monkeypatch):
    _build_only(_config(monkeypatch, fleet=True))
    data = _data(customers=3, products=1, periods=2)
    data["aggregate_vehicles"] = True
    totals = np.zeros((1, 4, 2))
    totals[0, 1, 0] = 3.0
    totals[0, 3, 0] = 2.0
    fixed = {"X": np.full((1, 2), 5.0), "Y": np.ones((1, 2)), "delivery_totals": totals}
    model = MPPRP(data, "/tmp", Mock(), {"start": False}, fixed_plan=fixed)
    try:
        model.solver(timeLimit=1, numThreads=1)
        assert model.model.get_constraint_by_name("FIX_VISIT_1_0") is not None
        assert model.model.get_constraint_by_name("FIX_VISIT_3_0") is not None
        assert model.model.get_constraint_by_name("FIX_VISIT_2_0") is None
        assert model.model.get_constraint_by_name("FIX_VISIT_1_1") is None
        assert model.model.get_constraint_by_name("FIX_X_0_0") is not None
        assert model.model.get_constraint_by_name("FIX_Q_0_1_0") is not None
        # customer 2 has no delivery: no arc touches it in period 0
        assert not any(
            key[1] == 2 or key[2] == 2 for key in model.model.Z_v_i_k_t if key[3] == 0
        )
    finally:
        model.terminate()


def test_vehicle_free_stage_models_are_smaller(monkeypatch):
    _config(monkeypatch)
    data = _data(customers=6, vehicles=4, periods=3)
    sizes = {}
    for fleet in (False, True):
        _build_only(_config(monkeypatch, fleet=fleet))
        wrapper = _matheuristic(dict(data))
        stage = MPPRP(
            wrapper.data,
            "/tmp",
            Mock(),
            {"start": False},
            routing_profile=(
                None if fleet else {"mode": "ordered", "order": wrapper.order}
            ),
            arc_order=wrapper.order if fleet else None,
        )
        try:
            stage.solver(timeLimit=1, numThreads=1)
            sizes[fleet] = stage.model.number_of_variables
        finally:
            stage.terminate()
    assert sizes[True] < sizes[False] / 2


# ----------------------------------------------------------- CPLEX integration


def _solve_stage(monkeypatch, data, fleet, multiobjective):
    _config(monkeypatch, fleet=fleet, multiobjective=multiobjective)
    wrapper = DeterministicMatheuristic(dict(data), "/tmp", Mock())
    wrapper.order = [0, 1, 2, 3, 4, 5, 0]
    start = wrapper._constructive_start()
    restricted = wrapper._stage(
        "restricted", start, 30, 2, {"mode": "ordered", "order": wrapper.order}
    )
    return wrapper, start, restricted


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
@pytest.mark.parametrize("multiobjective", [False, True])
def test_vehicle_free_restricted_stage_matches_the_vehicle_indexed_optimum(
    monkeypatch, multiobjective
):
    data = _data(customers=5, products=2, vehicles=2, periods=3, capacity=30, seed=3)
    results = {}
    for fleet in (False, True):
        wrapper, start, restricted = _solve_stage(
            monkeypatch, data, fleet, multiobjective
        )
        assert restricted is not None
        record = wrapper.stages[-1]
        assert "optimal" in record["status"].lower()
        problem = ProblemData.from_map(data)
        assert validate_variables(problem, restricted, ordered_arcs(wrapper.order))[
            "feasible"
        ]
        assert wrapper._score(restricted)[0] <= wrapper._score(start)[0] + 1e-8
        results[fleet] = (wrapper._score(restricted)[0], record["variables"])
    assert results[True][0] == pytest.approx(results[False][0], rel=1e-6, abs=1e-6)
    assert results[True][1] < results[False][1]


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
def test_vehicle_free_route_improvement_keeps_the_plan_and_never_worsens(monkeypatch):
    data = _data(customers=5, products=2, vehicles=2, periods=3, capacity=30, seed=5)
    wrapper, _, restricted = _solve_stage(monkeypatch, data, True, True)
    fixed = {
        "X": restricted["X"],
        "Y": restricted["Y"],
        "delivery_totals": wrapper._delivery_totals(restricted),
    }
    improved = wrapper._stage(
        "route_improvement",
        wrapper._improvement_start(restricted),
        15,
        2,
        {"mode": "improvement"},
        fixed,
    )
    assert improved is not None
    np.testing.assert_allclose(improved["X"], restricted["X"], atol=1e-6)
    np.testing.assert_allclose(improved["Y"], restricted["Y"])
    np.testing.assert_allclose(
        improved["Q"].sum(axis=1), fixed["delivery_totals"], atol=1e-6
    )
    problem = ProblemData.from_map(data)
    assert validate_variables(problem, improved)["feasible"]
    assert wrapper._score(improved)[0] <= wrapper._score(restricted)[0] + 1e-8
    assert wrapper.stages[-1]["gap_scope"] == "route_improvement"


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
@pytest.mark.parametrize("fleet", [False, True])
def test_matheuristic_only_flow_returns_a_valid_solution(monkeypatch, fleet):
    data = _data(customers=5, products=2, vehicles=2, periods=3, capacity=30, seed=7)
    _config(monkeypatch, fleet=fleet, start="pso", multiobjective=True)
    wrapper = DeterministicMatheuristic(dict(data), "/tmp", Mock())
    wrapper.solver(numThreads=2)
    assert wrapper.best is not None
    problem = ProblemData.from_map(data)
    assert validate_variables(problem, wrapper.best)["feasible"]
    names = [stage["stage"] for stage in wrapper.stages]
    assert names == ["tsp", "restricted", "route_improvement"]
    assert wrapper.selected_stage in ("restricted", "route_improvement")
