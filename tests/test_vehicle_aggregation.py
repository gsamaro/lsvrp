"""Vehicle-free (aggregated) formulation: Z, R and Q lose the vehicle index.

The unit tests assemble models without solving; the CPLEX integration tests
solve small instances with and without the aggregation and compare optima.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from config import Config
from src.solvers._routing_common import (
    aggregate_vehicle_axis,
    expand_aggregated_vehicles,
    routes_from_z,
    validate_variables,
)
from src.solvers._solver_common import ProblemData, evaluate_configured_objective
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)

HAS_CPLEX = importlib.util.find_spec("cplex") is not None


def _instance(customers=3, vehicles=3, periods=2, capacity=5, demand=2, stock=50):
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


@pytest.fixture
def configured(monkeypatch):
    value = Config.normalize(
        {
            "solver": {"multiobjective": False, "method": "GUROBY"},
            "relaxed_solution": {"use": True},
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    return value


def _build(data, **kwargs):
    """Assemble the whole model through ``solver()`` without solving it."""
    solver = MPPRP(data, "/tmp", Mock(), {"start": False}, **kwargs)
    solver.solver(timeLimit=1, numThreads=1)
    return solver


# --------------------------------------------------------------------------- config


def test_config_default_is_off_and_validated():
    assert Config.normalize({})["solver"]["aggregate_vehicles"] is False
    assert (
        Config.normalize({"solver": {"aggregate_vehicles": True}})["solver"][
            "aggregate_vehicles"
        ]
        is True
    )
    with pytest.raises(ValueError, match="aggregate_vehicles"):
        Config.normalize({"solver": {"aggregate_vehicles": "true"}})


def test_shipped_config_declares_the_flag():
    path = Path(__file__).resolve().parents[1] / "config" / "config.json"
    shipped = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(shipped["solver"]["aggregate_vehicles"], bool)


# ------------------------------------------------------------------ flag resolution


@pytest.mark.parametrize(
    ("configured_value", "map_value", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (False, True, True),
        (True, False, False),
    ],
)
def test_data_map_takes_precedence_over_config(
    configured, configured_value, map_value, expected
):
    configured["solver"]["aggregate_vehicles"] = configured_value
    data = _instance()
    if map_value is not None:
        data["aggregate_vehicles"] = map_value
    solver = MPPRP(data, "/tmp", Mock(), {"start": False})
    assert solver.aggregate_vehicles is expected
    assert solver.v_model == (1 if expected else data["num_vehicles"])
    assert solver.departure_limit == (data["num_vehicles"] if expected else 1)
    solver.terminate()


def _fixed_plan():
    return {
        "X": np.zeros((1, 2)),
        "Y": np.zeros((1, 2)),
        "delivery_totals": np.zeros((1, 4, 2)),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_profile": {"mode": "improvement"}},
        {"routing_profile": {"mode": "improvement"}, "fixed_plan": _fixed_plan()},
    ],
)
def test_routing_profiles_keep_per_vehicle_assignment(configured, kwargs):
    data = _instance()
    data["aggregate_vehicles"] = True
    solver = MPPRP(data, "/tmp", Mock(), {"start": False}, **kwargs)
    assert solver.aggregate_vehicles is False
    assert solver.v_model == data["num_vehicles"]
    solver.terminate()


@pytest.mark.parametrize(
    "kwargs",
    [{"fixed_plan": _fixed_plan()}, {"arc_order": [0, 1, 2, 3, 0]}],
)
def test_fixed_plan_and_arc_order_can_be_aggregated(configured, kwargs):
    data = _instance()
    data["aggregate_vehicles"] = True
    solver = MPPRP(data, "/tmp", Mock(), {"start": False}, **kwargs)
    assert solver.aggregate_vehicles is True
    assert solver.v_model == 1
    solver.terminate()


@pytest.mark.parametrize(
    ("target_use_relaxation", "expected"), [(True, False), (False, True)]
)
def test_relaxed_target_construction_is_never_aggregated(
    configured, target_use_relaxation, expected
):
    # The relaxation exports fractional arcs, which cannot be split into routes.
    configured["solver"]["aggregate_vehicles"] = True
    configured["postprocessing"]["build_target"] = True
    configured["relaxed_solution"]["target_use_relaxation"] = target_use_relaxation
    solver = MPPRP(_instance(), "/tmp", Mock(), {"start": False})
    assert solver.aggregate_vehicles is expected
    solver.terminate()


# --------------------------------------------------------------------- model shape


def test_aggregated_model_has_one_fleet_slot(configured):
    customers, vehicles, periods = 3, 3, 2
    arcs = (customers + 1) * customers
    indexed = _build(_instance(customers, vehicles, periods))
    aggregated = _build(
        {**_instance(customers, vehicles, periods), "aggregate_vehicles": True}
    )
    try:
        assert len(indexed.model.Z_v_i_k_t) == vehicles * arcs * periods
        assert len(aggregated.model.Z_v_i_k_t) == arcs * periods
        assert len(aggregated.model.R_p_v_i_k_t) == arcs * periods
        assert len(aggregated.model.Q_p_v_i_t) == customers * periods
        assert {key[0] for key in aggregated.model.Z_v_i_k_t} == {0}
        assert aggregated.model.number_of_binary_variables == (
            arcs * periods + aggregated.p * periods
        )
        assert aggregated.applied_features["aggregate_vehicles"] is True
        assert indexed.applied_features["aggregate_vehicles"] is False
    finally:
        indexed.terminate()
        aggregated.terminate()


def test_departures_are_bounded_by_the_fleet_size(configured):
    vehicles = 3
    indexed = _build(_instance(vehicles=vehicles))
    aggregated = _build({**_instance(vehicles=vehicles), "aggregate_vehicles": True})
    try:
        indexed_limit = indexed.model.get_constraint_by_name("EQ_10_v_2_t_0")
        fleet_limit = aggregated.model.get_constraint_by_name("EQ_10_v_0_t_0")
        assert indexed_limit.rhs.constant == 1
        assert fleet_limit.rhs.constant == vehicles
        assert aggregated.model.get_constraint_by_name("EQ_10_v_1_t_0") is None
    finally:
        indexed.terminate()
        aggregated.terminate()


def test_aggregation_drops_symmetry_breaking_and_keeps_coelho(configured):
    solver = _build(
        {**_instance(), "aggregate_vehicles": True},
        symmetry_breaking_hc1=True,
        coelho_inequalities=True,
    )
    try:
        assert solver.applied_features["hc1"] is False
        assert solver.applied_features["coelho_inequalities"] is True
        names = [constraint.name for constraint in solver.model.iter_constraints()]
        assert not any(name.startswith("SB_") for name in names)
        assert "COELHO_17_v_0_i_1_t_0" in names
        assert not any("COELHO_17_v_1" in name for name in names)
    finally:
        solver.terminate()


# ------------------------------------------------------------------ route splitting


def _problem_and_aggregated_solution():
    data = _instance(customers=3, vehicles=3, periods=2, capacity=5, demand=0)
    data["d_pit"] = [[[2, 0], [1, 0], [4, 0]]]
    problem = ProblemData.from_map(data)
    q = np.zeros((1, 1, 4, 2))
    q[0, 0, 1:, 0] = [2, 1, 4]
    variables = {
        "X": np.array([[7.0, 0.0]]),
        "Y": np.array([[1.0, 0.0]]),
        "I": np.zeros((1, 4, 2)),
        "Q": q,
        # Fleet-wide slot: routes 0-1-2-0 (load 3) and 0-3-0 (load 4).
        "Z": {
            (0, 0, 1, 0): 1.0,
            (0, 1, 2, 0): 1.0,
            (0, 2, 0, 0): 1.0,
            (0, 0, 3, 0): 1.0,
            (0, 3, 0, 0): 1.0,
        },
        "R": {(0, 0, 0, 1, 0): 3.0, (0, 0, 1, 2, 0): 1.0, (0, 0, 0, 3, 0): 4.0},
    }
    return problem, variables


def test_expansion_assigns_one_vehicle_per_route():
    problem, aggregated = _problem_and_aggregated_solution()
    expanded = expand_aggregated_vehicles(aggregated, problem.v)

    assert expanded["Q"].shape == (1, 3, 4, 2)
    # Canonical labels: the route holding customer 1 is vehicle 0.
    assert routes_from_z(expanded["Z"], shape=(3, 4, 4, 2))[0] == [[1, 2], [3], []]
    np.testing.assert_allclose(expanded["Q"][0, 0, :, 0], [0, 2, 1, 0])
    np.testing.assert_allclose(expanded["Q"][0, 1, :, 0], [0, 0, 0, 4])
    assert expanded["R"] == {
        (0, 0, 0, 1, 0): 3.0,
        (0, 0, 1, 2, 0): 1.0,
        (0, 1, 0, 3, 0): 4.0,
    }
    # Quantities are only redistributed among vehicles.
    np.testing.assert_allclose(expanded["Q"].sum(axis=1), aggregated["Q"].sum(axis=1))
    assert validate_variables(problem, expanded)["feasible"]


def test_expanded_routes_respect_vehicle_capacity():
    problem, aggregated = _problem_and_aggregated_solution()
    expanded = expand_aggregated_vehicles(aggregated, problem.v)
    loads = expanded["Q"][:, :, :, 0].sum(axis=(0, 2))
    assert loads.tolist() == [3.0, 4.0, 0.0]
    assert np.all(loads <= problem.C)
    # A route above C is not hidden by the split: the audit rejects it.
    expanded["Q"][0, 1, 3, 0] = 6.0
    assert not validate_variables(problem, expanded)["feasible"]


def test_expansion_rejects_more_routes_than_vehicles():
    problem, aggregated = _problem_and_aggregated_solution()
    with pytest.raises(ValueError, match="excedem 1 veículos"):
        expand_aggregated_vehicles(aggregated, 1)


def test_expansion_is_stable_under_relabelling_of_customers_in_route_order():
    # Routes are labelled by their smallest customer, not by arc order.
    problem, aggregated = _problem_and_aggregated_solution()
    reordered = dict(aggregated)
    reordered["Z"] = dict(reversed(list(aggregated["Z"].items())))
    expanded = expand_aggregated_vehicles(reordered, problem.v)
    assert routes_from_z(expanded["Z"], shape=(3, 4, 4, 2))[0] == [[1, 2], [3], []]


def test_disconnected_cycle_stays_on_vehicle_zero_and_is_rejected():
    problem, aggregated = _problem_and_aggregated_solution()
    aggregated["Z"][(0, 1, 2, 1)] = 1.0
    aggregated["Z"][(0, 2, 1, 1)] = 1.0
    expanded = expand_aggregated_vehicles(aggregated, problem.v)
    assert expanded["Z"][(0, 1, 2, 1)] == expanded["Z"][(0, 2, 1, 1)] == 1.0
    assert not validate_variables(problem, expanded)["feasible"]


def test_aggregate_vehicle_axis_sums_dense_and_sparse_starts():
    z = np.zeros((2, 3, 3, 1))
    z[0, 0, 1, 0] = 1
    z[1, 0, 2, 0] = 1
    assert aggregate_vehicle_axis("Z", z).shape == (3, 3, 1)
    assert aggregate_vehicle_axis("Z", z)[0, 2, 0] == 1
    q = np.zeros((1, 2, 3, 1))
    q[0, 0, 1, 0], q[0, 1, 2, 0] = 2.0, 3.0
    assert aggregate_vehicle_axis("Q", q)[0, :, 0].tolist() == [0.0, 2.0, 3.0]
    sparse = {(0, 1, 0, 1, 0): 2.0, (0, 0, 0, 1, 0): 1.5}
    assert aggregate_vehicle_axis("R", sparse) == {(0, 0, 1, 0): 3.5}


def test_mip_start_from_vehicle_indexed_solution_is_summed(configured):
    data = _instance(customers=3, vehicles=2, periods=1, demand=1)
    data["aggregate_vehicles"] = True
    q = np.zeros((1, 2, 4, 1))
    q[0, 0, 1, 0], q[0, 1, 3, 0] = 1.0, 1.0
    z = np.zeros((2, 4, 4, 1))
    z[0, 0, 1, 0] = z[0, 1, 0, 0] = 1
    z[1, 0, 3, 0] = z[1, 3, 0, 0] = 1
    r = np.zeros((1, 2, 4, 4, 1))
    r[0, 0, 0, 1, 0] = 1.0
    r[0, 1, 0, 3, 0] = 1.0
    start = {
        "X": [[2.0]],
        "Y": [[1]],
        "I": np.zeros((1, 4, 1)),
        "Q": q,
        "R": r,
        "Z": z,
    }
    solver = MPPRP(data, "/tmp", Mock(), {"start": True, "variables": start})
    solver.createDecisionVariables()
    solver.startVariables()

    warm_start = next(iter(solver.model.iter_mip_starts()))[0]
    # Both vehicles' arcs now live in the single fleet-wide slot.
    assert warm_start.get_value(solver.model.Z_v_i_k_t[0, 0, 1, 0]) == 1
    assert warm_start.get_value(solver.model.Z_v_i_k_t[0, 0, 3, 0]) == 1
    assert warm_start.get_value(solver.model.Z_v_i_k_t[0, 3, 0, 0]) == 1
    assert warm_start.get_value(solver.model.Z_v_i_k_t[0, 1, 2, 0]) == 0
    assert warm_start.get_value(solver.model.Y_p_t[0, 0]) == 1
    solver.terminate()


# ---------------------------------------------------------------- CPLEX equivalence


def _force_two_routes():
    """Four customers, C=7 and U=3: one route cannot serve a whole period."""
    data = _instance(customers=4, vehicles=2, periods=3, capacity=7, demand=3, stock=3)
    data["d_pit"] = [[[3] * 3 for _ in range(4)]]
    return data


def _solve(monkeypatch, data, aggregate, multiobjective, start=None, options=None):
    Config.load()
    configured = Config.normalize(
        {
            "solver": {
                "multiobjective": multiobjective,
                "method": "GUROBY",
                "aggregate_vehicles": aggregate,
                **(options or {}),
            },
            "relaxed_solution": {"use": False},
        }
    )
    monkeypatch.setattr(Config, "_data", configured)
    solver = MPPRP(dict(data), "/tmp", Mock(), start or {"start": False})
    solver.solver(timeLimit=60, numThreads=2)
    return solver


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
@pytest.mark.parametrize("multiobjective", [False, True])
def test_aggregated_optimum_matches_vehicle_indexed_optimum(
    monkeypatch, multiobjective
):
    data = _force_two_routes()
    problem = ProblemData.from_map(data)
    results = {}
    for aggregate in (False, True):
        solver = _solve(monkeypatch, data, aggregate, multiobjective)
        try:
            assert "optimal" in str(solver.model.solve_details.status).lower()
            variables = solver.get_variables()
            results[aggregate] = (float(solver.model.objective_value), variables)
        finally:
            solver.terminate()

    assert results[True][0] == pytest.approx(results[False][0], rel=1e-6, abs=1e-6)

    variables = results[True][1]
    assert validate_variables(problem, variables)["feasible"]
    routes = routes_from_z(variables["Z"])
    assert max(sum(bool(route) for route in period) for period in routes) == 2
    loads = variables["Q"].sum(axis=(0, 2))  # vehicle x period
    assert np.all(loads <= problem.C + 1e-6)
    # The split solution costs exactly what the model reported.
    assert evaluate_configured_objective(
        problem, variables, Config.snapshot()
    ) == pytest.approx(results[True][0], rel=1e-6, abs=1e-6)


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
def test_aggregated_model_accepts_a_vehicle_indexed_mip_start(monkeypatch):
    data = _force_two_routes()
    indexed = _solve(monkeypatch, data, aggregate=False, multiobjective=False)
    try:
        variables = indexed.get_variables()
        objective = float(indexed.model.objective_value)
    finally:
        indexed.terminate()

    start = {
        "start": True,
        "variables": {name: variables[name] for name in "XYIQRZ"},
    }
    aggregated = _solve(
        monkeypatch, data, aggregate=True, multiobjective=False, start=start
    )
    try:
        assert aggregated.model.number_of_mip_starts == 1
        assert "optimal" in str(aggregated.model.solve_details.status).lower()
        assert float(aggregated.model.objective_value) == pytest.approx(
            objective, rel=1e-6, abs=1e-6
        )
    finally:
        aggregated.terminate()


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
def test_optional_features_keep_the_aggregated_optimum(monkeypatch):
    data = _force_two_routes()
    baseline = _solve(monkeypatch, data, aggregate=False, multiobjective=False)
    try:
        expected = float(baseline.model.objective_value)
    finally:
        baseline.terminate()

    options = {
        "coelho_inequalities": True,
        "symmetry_breaking": {"hc1": True},
        "rounded_capacity_inequalities": {
            "enabled": True,
            "node_frequency": 1,
            "max_non_root_node": 5,
        },
    }
    solver = _solve(
        monkeypatch, data, aggregate=True, multiobjective=False, options=options
    )
    try:
        assert solver.applied_features["coelho_inequalities"] is True
        assert solver.applied_features["hc1"] is False
        assert solver.applied_features["rounded_capacity_callback"] is True
        assert "optimal" in str(solver.model.solve_details.status).lower()
        assert float(solver.model.objective_value) == pytest.approx(
            expected, rel=1e-6, abs=1e-6
        )
    finally:
        solver.terminate()
