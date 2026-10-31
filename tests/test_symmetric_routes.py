"""Undirected routes (``solver.symmetric_routes``) and the f1 switch (``optimize_f1``).

The distances are symmetric, so the model can use one variable per edge instead of
two arcs. The unit tests build models and helpers without solving; the CPLEX
integration tests solve small instances with and without the symmetry and compare
optima, check the conversions of starts and routes, and the f1 switch.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from config import Config
from src.solvers._routing_common import (
    directed_routes,
    edge_walks,
    validate_variables,
)
from src.solvers._solver_common import (
    ProblemData,
    effective_goal_weights,
    minimum_production_totals,
    objective_from_components,
    solution_components,
    trim_surplus,
)
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)

HAS_CPLEX = importlib.util.find_spec("cplex") is not None
needs_cplex = pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX not installed")

COORDINATES = [(0, 0), (4, 0), (8, 0), (0, 4), (0, 8), (5, 5)]


def _instance(customers=3, vehicles=3, periods=2, capacity=5, demand=2, stock=50):
    nodes = customers + 1
    coordinates = COORDINATES[:nodes]
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


def _configure(monkeypatch, multiobjective=False, symmetric=False, optimize_f1=True,
               **solver):
    value = Config.normalize(
        {
            "solver": {
                "multiobjective": multiobjective,
                "method": "GUROBY",
                "aggregate_vehicles": True,
                "symmetric_routes": symmetric,
                "goal_programming": {"optimize_f1": optimize_f1},
                **solver,
            },
            "relaxed_solution": {"use": False},
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    return value


def _solve(data, start=None, time_limit=30, **kwargs):
    solver = MPPRP(
        data, "/tmp", Mock(), start or {"start": False}, **kwargs
    )
    solver.solver(timeLimit=time_limit, numThreads=1)
    return solver


# ------------------------------------------------------------------------- config


def test_defaults_keep_the_previous_behaviour():
    solver = Config.normalize({})["solver"]
    assert solver["symmetric_routes"] is False
    assert solver["goal_programming"]["optimize_f1"] is True


def test_symmetric_routes_needs_aggregated_vehicles_and_booleans():
    with pytest.raises(ValueError, match="symmetric_routes"):
        Config.normalize({"solver": {"symmetric_routes": "yes"}})
    with pytest.raises(ValueError, match="aggregate_vehicles"):
        Config.normalize({"solver": {"symmetric_routes": True}})
    with pytest.raises(ValueError, match="optimize_f1"):
        Config.normalize({"solver": {"goal_programming": {"optimize_f1": 1}}})
    ok = Config.normalize(
        {"solver": {"symmetric_routes": True, "aggregate_vehicles": True}}
    )
    assert ok["solver"]["symmetric_routes"] is True


def test_shipped_config_declares_both_flags():
    path = Path(__file__).resolve().parents[1] / "config" / "config.json"
    solver = json.loads(path.read_text(encoding="utf-8"))["solver"]
    assert solver["symmetric_routes"] is False
    assert solver["goal_programming"]["optimize_f1"] is True


# ----------------------------------------------------------------- flag resolution


def _fixed_plan():
    return {
        "X": np.zeros((1, 2)),
        "Y": np.zeros((1, 2)),
        "delivery_totals": np.zeros((1, 4, 2)),
    }


def test_unrestricted_model_is_symmetric_and_fleet_wide(monkeypatch):
    _configure(monkeypatch, symmetric=True)
    solver = MPPRP(_instance(), "/tmp", Mock(), {"start": False})
    assert solver.symmetric_routes is True
    assert solver.aggregate_vehicles is True
    assert solver.v_model == 1 and solver.departure_limit == 3
    solver.terminate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_profile": {"mode": "improvement"}},
        {"fixed_plan": _fixed_plan()},
        {"arc_order": [0, 1, 2, 3, 0]},
    ],
)
def test_matheuristic_stages_keep_directed_arcs(monkeypatch, kwargs):
    _configure(monkeypatch, symmetric=True)
    solver = MPPRP(_instance(), "/tmp", Mock(), {"start": False}, **kwargs)
    assert solver.symmetric_routes is False
    solver.terminate()


def test_data_map_takes_precedence_over_config(monkeypatch):
    _configure(monkeypatch, symmetric=False)
    data = {**_instance(), "symmetric_routes": True}
    solver = MPPRP(data, "/tmp", Mock(), {"start": False})
    assert solver.symmetric_routes is True
    solver.terminate()


def test_asymmetric_distances_are_refused(monkeypatch):
    _configure(monkeypatch, symmetric=True)
    data = _instance()
    data["a_ik"][1][2] += 1
    with pytest.raises(ValueError, match="simétrica"):
        MPPRP(data, "/tmp", Mock(), {"start": False})


# --------------------------------------------------------------------- model shape


@needs_cplex
def test_symmetric_model_has_one_variable_per_edge(monkeypatch):
    customers, periods = 3, 2
    edges = (customers + 1) * customers // 2
    _configure(monkeypatch, symmetric=True)
    solver = _solve(_instance(customers=customers, periods=periods), time_limit=1)
    try:
        assert len(solver.model.Xe_i_k_t) == edges * periods
        assert len(solver.model.Ue_i_k_t) == edges * periods
        assert len(solver.model.Ye_i_t) == customers * periods
        assert len(solver.model.Z_v_i_k_t) == len(solver.model.R_p_v_i_k_t) == 0
        # Plant edges may be crossed twice (a route with a single customer).
        assert solver.model.Xe_i_k_t[0, 1, 0].ub == 2
        assert solver.model.Xe_i_k_t[1, 2, 0].ub == 1
        assert solver.applied_features["symmetric_routes"] is True
        names = {c.name or "" for c in solver.model.iter_constraints()}
        assert "SYM_DEPARTURES_t_0" in names and "SYM_DEGREE_k_1_t_0" in names
        assert "COELHO_16_i_1_k_2_node_2_t_0" in names
        assert solver.model.get_constraint_by_name("SYM_DEPARTURES_t_0").rhs.constant == 6
    finally:
        solver.terminate()


@needs_cplex
def test_symmetric_model_is_smaller_than_the_directed_one(monkeypatch):
    data = _instance(customers=5, vehicles=2, periods=3)
    _configure(monkeypatch, symmetric=False)
    directed = _solve(dict(data), time_limit=1)
    _configure(monkeypatch, symmetric=True)
    symmetric = _solve(dict(data), time_limit=1)
    try:
        assert symmetric.model.number_of_variables < directed.model.number_of_variables
        assert symmetric.model.number_of_binary_variables + (
            symmetric.model.number_of_integer_variables
        ) < directed.model.number_of_binary_variables
    finally:
        directed.terminate()
        symmetric.terminate()


# ------------------------------------------------------------------- route walking


def test_edge_walks_orient_routes_and_find_singletons_and_cycles():
    edges = {
        (0, 3): 1, (1, 3): 1, (0, 1): 1,  # route 0-3-1-0
        (0, 2): 2,  # single-customer route
        (4, 5): 1, (4, 6): 1, (5, 6): 1,  # loop detached from the plant
        (1, 2): 0.2,  # fractional noise below the threshold
    }
    routes, cycles = edge_walks(edges)
    assert sorted(routes) == [[1, 3], [2]]  # first customer has the smaller index
    assert len(cycles) == 1 and sorted(cycles[0]) == [4, 5, 6]


def test_directed_routes_follow_the_successors():
    arcs = {(0, 2): 1.0, (2, 1): 1.0, (1, 0): 1.0, (0, 3): 1.0, (3, 0): 1.0}
    assert sorted(directed_routes(arcs)) == [[2, 1], [3]]


# ------------------------------------------------------------- equivalence (CPLEX)


@needs_cplex
@pytest.mark.parametrize("capacity", [5, 7, 20])
@pytest.mark.parametrize("multiobjective", [False, True])
def test_symmetric_and_directed_models_share_the_optimum(
    monkeypatch, capacity, multiobjective
):
    data = _instance(customers=4, vehicles=2, periods=3, capacity=capacity)
    results = {}
    for symmetric in (False, True):
        _configure(monkeypatch, multiobjective=multiobjective, symmetric=symmetric)
        solver = _solve(dict(data))
        problem = ProblemData.from_map(data)
        variables = solver.get_variables()
        assert validate_variables(problem, variables)["feasible"]
        results[symmetric] = (
            solver.model.objective_value,
            solution_components(problem, variables).sum(),
        )
        solver.terminate()
    assert results[True][0] == pytest.approx(results[False][0], rel=1e-6)
    if not multiobjective:
        # Several plans can tie on the goal criterion, but not on the total cost.
        assert results[True][1] == pytest.approx(results[False][1], rel=1e-6)


@needs_cplex
def test_route_load_never_exceeds_the_vehicle_capacity(monkeypatch):
    # Demand 2 per customer: three customers (6) do not fit a capacity of 5.
    data = _instance(customers=4, vehicles=4, periods=2, capacity=5, demand=2)
    _configure(monkeypatch, symmetric=True)
    solver = _solve(data)
    try:
        variables = solver.get_variables()
        loads = variables["Q"].sum(axis=(0, 2))  # vehicle x period
        assert np.all(loads <= 5 + 1e-6)
        assert validate_variables(ProblemData.from_map(data), variables)["feasible"]
    finally:
        solver.terminate()


# ----------------------------------------------------------------- start conversion


def _capture_start(monkeypatch):
    import docplex.mp.model as docplex_model

    captured = {}
    original = docplex_model.Model.add_mip_start

    def spy(self, solution, *args, **kwargs):
        captured["valid"] = solution.is_valid_solution(tolerance=1e-5)
        captured["solution"] = solution
        return original(self, solution, *args, **kwargs)

    monkeypatch.setattr(docplex_model.Model, "add_mip_start", spy)
    return captured


@needs_cplex
@pytest.mark.parametrize("multiobjective", [False, True])
def test_a_directed_solution_is_a_valid_start_of_the_symmetric_model(
    monkeypatch, multiobjective
):
    data = _instance(customers=4, vehicles=2, periods=3, capacity=5)
    _configure(monkeypatch, multiobjective=multiobjective, symmetric=False)
    directed = _solve(dict(data))
    variables = directed.get_variables()
    objective = directed.model.objective_value
    directed.terminate()

    _configure(monkeypatch, multiobjective=multiobjective, symmetric=True)
    captured = _capture_start(monkeypatch)
    symmetric = _solve(
        dict(data), start={"start": True, "variables": variables}, time_limit=5
    )
    try:
        assert captured["valid"] is True
        assert symmetric.model.objective_value == pytest.approx(objective, rel=1e-6)
    finally:
        symmetric.terminate()


@needs_cplex
def test_a_start_with_a_wrong_edge_flow_is_not_valid(monkeypatch):
    data = _instance(customers=3, vehicles=2, periods=2, capacity=5)
    _configure(monkeypatch, symmetric=False)
    directed = _solve(dict(data))
    variables = directed.get_variables()
    directed.terminate()
    # Dropping a delivery breaks the flow conservation of the converted start.
    variables["Q"] = np.array(variables["Q"])
    variables["Q"][0, :, 1, 0] += 1.0

    _configure(monkeypatch, symmetric=True)
    captured = _capture_start(monkeypatch)
    solver = _solve(dict(data), start={"start": True, "variables": variables}, time_limit=2)
    try:
        assert captured["valid"] is False
    finally:
        solver.terminate()


# -------------------------------------------------------- relaxed target construction


@needs_cplex
def test_relaxed_targets_with_symmetric_routes_keep_the_transport_costs(monkeypatch):
    data = _instance(customers=4, vehicles=2, periods=2, capacity=5)
    data["weight"] = [0, 0, 0, 1, 0]  # one of the five single-objective runs
    values = {}
    for symmetric in (False, True):
        config = Config.normalize(
            {
                "solver": {
                    "multiobjective": False,
                    "method": "GUROBY",
                    "aggregate_vehicles": symmetric,
                    "symmetric_routes": symmetric,
                },
                "postprocessing": {"build_target": True},
                "relaxed_solution": {
                    "use": False,
                    "replace_model": True,
                    "target_use_relaxation": True,
                },
            }
        )
        monkeypatch.setattr(Config, "_data", config)
        solver = _solve(dict(data))
        variables = solver.get_variables()
        components = solution_components(ProblemData.from_map(data), variables)
        values[symmetric] = (solver.relaxedModelObjVal, components.sum(axis=1))
        assert variables["Z"].shape == (2, 5, 5, 2)  # vehicle axis kept for outputs
        assert variables["Q"].shape[1] == 2
        solver.terminate()
    assert values[True][0] == pytest.approx(values[False][0], rel=1e-6)
    assert values[True][1][3] == pytest.approx(values[False][1][3], rel=1e-6)
    assert values[True][1][3] == pytest.approx(values[True][0], rel=1e-6)


# ------------------------------------------------------------------------ f1 switch


def test_effective_weights_drop_f1_and_sum_to_one():
    assert effective_goal_weights([0.2] * 5, True).tolist() == [0.2] * 5
    off = effective_goal_weights([0.2] * 5, False)
    assert off[0] == 0 and off[1:].tolist() == pytest.approx([0.25] * 4)
    off = effective_goal_weights([0.35, 0.35, 0.1, 0.1, 0.1], False)
    assert off.sum() == pytest.approx(1.0)
    assert off[1] == pytest.approx(0.35 / 0.65)
    with pytest.raises(ValueError, match="f2..f5"):
        effective_goal_weights([1, 0, 0, 0, 0], False)


def test_the_configured_objective_ignores_f1_when_it_is_off(monkeypatch):
    data = _instance()
    problem = ProblemData.from_map(data)
    components = np.array([[500.0, 500.0], [40.0, 40.0], [30.0, 30.0], [80.0, 80.0],
                           [60.0, 60.0]])
    on = {"solver": {"multiobjective": True}}
    off = {"solver": {"multiobjective": True, "goal_programming": {"optimize_f1": False}}}
    with_f1 = objective_from_components(problem, components, on)
    without_f1 = objective_from_components(problem, components, off)
    # With f1 far above its target the criterion with f1 is larger.
    assert with_f1 > without_f1 > 0
    cheaper_f1 = components.copy()
    cheaper_f1[0] = 0
    assert objective_from_components(problem, cheaper_f1, off) == pytest.approx(
        without_f1
    )


def test_minimum_production_is_the_demand_beyond_the_initial_stock():
    data = _instance(customers=2, periods=3, demand=4)
    data["I_pi0"] = [[5, 6, 20]]  # plant 5; customer 1 covers 6 of 12; customer 2 all
    problem = ProblemData.from_map(data)
    # customer 1 needs 6, customer 2 needs 0, the plant stock covers 5 of them.
    assert minimum_production_totals(problem).tolist() == [1.0]
    data["I_pi0"] = [[50, 0, 0]]
    assert minimum_production_totals(ProblemData.from_map(data)).tolist() == [0.0]


def _wasteful_plan():
    data = _instance(customers=3, periods=2, demand=2)
    problem = ProblemData.from_map(data)
    q = np.zeros((1, 1, 4, 2))
    q[0, 0, 1:, 0] = 3.0
    q[0, 0, 1:, 1] = 2.0
    inventory = np.zeros((1, 4, 2))
    inventory[0, 1:, :] = 1.0  # one unit left at every customer at the end
    inventory[0, 0, :] = [0.0, 6.0]
    plan = {
        "X": np.array([[9.0, 9.0]]),
        "Y": np.array([[1.0, 1.0]]),
        "I": inventory,
        "Q": q,
    }
    return problem, plan


def test_trim_surplus_removes_the_final_stock_with_the_latest_flows():
    problem, plan = _wasteful_plan()
    trimmed = trim_surplus(problem, plan)

    assert trimmed["X"].sum() == pytest.approx(minimum_production_totals(problem)[0])
    assert trimmed["X"].tolist() == [[9.0, 3.0]]  # the latest production is cut
    assert trimmed["Q"][0, 0, 1:, 1].tolist() == [1.0, 1.0, 1.0]
    assert trimmed["Q"][0, 0, 1:, 0].tolist() == [3.0, 3.0, 3.0]
    assert np.all(trimmed["I"] >= 0)
    assert trimmed["I"][0, 1:, -1].tolist() == [0.0, 0.0, 0.0]
    assert plan["X"].tolist() == [[9.0, 9.0]]  # the input is not modified
    # The trimmed plan still balances.
    previous = np.concatenate(
        (np.zeros((1, 4, 1)), trimmed["I"][:, :, :-1]), axis=2
    )
    plant = previous[:, 0] + trimmed["X"] - trimmed["Q"][:, :, 1:].sum(axis=(1, 2))
    assert np.allclose(plant, trimmed["I"][:, 0])
    customers = previous[:, 1:] + trimmed["Q"].sum(axis=1)[:, 1:] - problem.d_p_i_t
    assert np.allclose(customers, trimmed["I"][:, 1:])


def test_trim_surplus_leaves_a_plan_without_waste_untouched():
    problem, plan = _wasteful_plan()
    trimmed = trim_surplus(problem, plan)
    assert trim_surplus(problem, trimmed) is trimmed


@needs_cplex
def test_with_f1_off_the_production_total_is_the_minimum(monkeypatch):
    data = _instance(customers=4, vehicles=2, periods=3, capacity=5)
    data["targets"] = [{f"f{j}_target": 1.0 for j in range(1, 6)}] * 3  # f3 free
    problem = ProblemData.from_map(data)
    _configure(monkeypatch, multiobjective=True, symmetric=True, optimize_f1=False)
    solver = _solve(data)
    try:
        assert solver.applied_features["optimize_f1"] is False
        assert solver.applied_features["minimum_production"] is True
        variables = solver.get_variables()
        assert variables["X"].sum() == pytest.approx(
            minimum_production_totals(problem).sum(), abs=1e-6
        )
        assert validate_variables(problem, variables)["feasible"]
        # p1 is reported from the plan, not left at whatever the model chose.
        result = solver.getResults()
        p1 = [row[0] for row in result[6]]
        f1 = solution_components(problem, variables)[0]
        expected = np.maximum(0.0, f1 - np.array([1.0] * 3))
        assert p1 == pytest.approx(expected.tolist())
    finally:
        solver.terminate()


@needs_cplex
def test_with_f1_on_nothing_is_fixed(monkeypatch):
    _configure(monkeypatch, multiobjective=True, symmetric=False, optimize_f1=True)
    solver = _solve(_instance(), time_limit=1)
    try:
        assert solver.applied_features["optimize_f1"] is True
        assert solver.applied_features["minimum_production"] is False
        names = {c.name or "" for c in solver.model.iter_constraints()}
        assert not any(name.startswith("MIN_PRODUCTION") for name in names)
    finally:
        solver.terminate()


@needs_cplex
def test_a_start_with_waste_is_trimmed_before_it_is_loaded(monkeypatch):
    data = _instance(customers=4, vehicles=2, periods=3, capacity=5)
    _configure(monkeypatch, multiobjective=True, symmetric=False, optimize_f1=False)
    first = _solve(dict(data))
    variables = first.get_variables()
    first.terminate()
    wasteful = {name: np.array(value) for name, value in variables.items()
                if isinstance(value, np.ndarray)}
    wasteful.update(Z=variables["Z"], R=variables["R"])
    wasteful["X"][0, 0] += 3.0  # three units produced and never used
    wasteful["I"][0, 0, :] += 3.0

    for symmetric in (False, True):
        _configure(monkeypatch, multiobjective=True, symmetric=symmetric,
                   optimize_f1=False)
        captured = _capture_start(monkeypatch)
        solver = _solve(dict(data), start={"start": True, "variables": wasteful},
                        time_limit=5)
        try:
            assert captured["valid"] is True
        finally:
            solver.terminate()


# ------------------------------------------------------- matheuristic hand-off (CPLEX)


def _matheuristic_data(customers=5, products=2, vehicles=2, periods=3, capacity=30):
    rng = np.random.default_rng(7)
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


@pytest.mark.cplex_integration
@needs_cplex
@pytest.mark.parametrize("optimize_f1", [True, False])
def test_matheuristic_hands_its_plan_to_the_symmetric_mip(monkeypatch, optimize_f1):
    from src.solvers.DeterministicMatheuristic import DeterministicMatheuristic

    data = _matheuristic_data()
    value = Config.normalize(
        {
            "solver": {
                "multiobjective": True,
                "method": "GUROBY",
                "aggregate_vehicles": True,
                "symmetric_routes": True,
                "goal_programming": {"optimize_f1": optimize_f1},
                "matheuristic": {
                    "enabled": True,
                    "start": "pso",
                    "use_as": "exact_start",
                    "tsp": {"time_limit": 5},
                    "restricted": {"time_limit": 20},
                    "route_improvement": {"enabled": False},
                },
                "pso": {
                    "swarm_size": 30,
                    "max_iterations": 30,
                    "execution_backend": "python",
                    "jit_warmup": False,
                    "lot_sizing_bounds": {"enabled": False},
                },
            },
            "relaxed_solution": {"use": False},
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    wrapper = DeterministicMatheuristic(dict(data), "/tmp", Mock())
    wrapper.solver(timeLimit=20, numThreads=1)

    problem = ProblemData.from_map(data)
    assert wrapper.best is not None
    assert validate_variables(problem, wrapper.best)["feasible"]
    stages = {stage["stage"]: stage for stage in wrapper.stages}
    exact = stages["exact_start"]
    assert exact["has_solution"] and not exact["validation_errors"]
    assert exact["features"]["symmetric_routes"] is True
    assert exact["features"]["optimize_f1"] is optimize_f1
    # The restricted stage keeps the directed arcs.
    assert stages["restricted"]["features"]["symmetric_routes"] is False
    if not optimize_f1:
        assert wrapper.best["X"].sum() == pytest.approx(
            minimum_production_totals(problem).sum(), abs=1e-6
        )


@pytest.mark.parametrize("optimize_f1", [True, False])
def test_the_pso_scores_particles_with_the_same_criterion(monkeypatch, optimize_f1):
    from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization

    _configure(
        monkeypatch,
        multiobjective=True,
        optimize_f1=optimize_f1,
        pso={
            "execution_backend": "python",
            "jit_warmup": False,
            "lot_sizing_bounds": {"enabled": False},
        },
    )
    pso = ParticleSwarmOptimization(_instance(), "/tmp", Mock())
    assert pso.objective_config["solver"]["goal_programming"] == {
        "optimize_f1": optimize_f1
    }
    problem = ProblemData.from_map(_instance())
    components = np.full((5, 2), 100.0)
    components[0] = 10_000.0  # f1 far above its target
    cost = objective_from_components(problem, components, pso.objective_config)
    reference = objective_from_components(
        problem, np.vstack([np.zeros((1, 2)), components[1:]]),
        pso.objective_config,
    )
    assert bool(cost == pytest.approx(reference)) is (not optimize_f1)
