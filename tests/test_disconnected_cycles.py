"""Cycles among customers detached from the plant in a MIP incumbent.

The model has no constraint tying an arc to the plant, so a cycle without
deliveries is feasible (and free when F5 is below its target). The solution
extracted from CPLEX drops those arcs before the route audit rejects them.
The unit tests build the real model without solving it and read a fake incumbent;
the CPLEX integration tests check that the model admits such cycles.
"""

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_vehicle_aggregation import _instance, _problem_and_aggregated_solution

from config import Config
from src.solvers._routing_common import (
    drop_disconnected_arcs,
    expand_aggregated_vehicles,
    routes_from_z,
    validate_variables,
)
from src.solvers._solver_common import (
    ProblemData,
    evaluate_configured_objective,
    goal_start_values,
    solution_components,
)
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)

HAS_CPLEX = importlib.util.find_spec("cplex") is not None

CYCLE = {(0, 1, 2, 1): 1.0, (0, 2, 1, 1): 1.0}
CYCLE_LOAD = {(0, 0, 1, 2, 1): 1.5, (0, 0, 2, 1, 1): 1.5}


def _with_cycle():
    problem, variables = _problem_and_aggregated_solution()
    variables["Z"].update(CYCLE)
    variables["R"].update(CYCLE_LOAD)
    return problem, variables


# ----------------------------------------------------------------------- the filter


def test_cycle_arcs_and_their_load_are_dropped_and_plant_routes_are_kept():
    problem, variables = _with_cycle()
    z, r, dropped = drop_disconnected_arcs(variables["Z"], variables["R"])

    assert dropped == set(CYCLE)
    assert set(z) == set(variables["Z"]) - set(CYCLE)
    assert set(r) == set(variables["R"]) - set(CYCLE_LOAD)
    # The input snapshot is not modified.
    assert set(CYCLE) <= set(variables["Z"])


def test_solution_without_cycles_is_returned_unchanged():
    _, variables = _problem_and_aggregated_solution()
    z, r, dropped = drop_disconnected_arcs(variables["Z"], variables["R"])
    assert dropped == set()
    assert z is variables["Z"] and r is variables["R"]


def test_cycle_is_judged_per_vehicle_and_period():
    # The same customers are on a plant route in period 0: only the period-1 cycle goes.
    _, variables = _with_cycle()
    _, _, dropped = drop_disconnected_arcs(variables["Z"], variables["R"])
    assert all(t == 1 for _, _, _, t in dropped)
    # A cycle of another vehicle slot is not rescued by a plant route of slot 0.
    z = {**variables["Z"], (1, 1, 2, 0): 1.0, (1, 2, 1, 0): 1.0}
    _, _, dropped = drop_disconnected_arcs(z, variables["R"])
    assert {(1, 1, 2, 0), (1, 2, 1, 0)} <= dropped


def test_longer_cycle_and_route_leaving_the_plant_through_it_are_distinguished():
    z = {
        (0, 0, 1, 0): 1.0,
        (0, 1, 0, 0): 1.0,
        (0, 2, 3, 0): 1.0,
        (0, 3, 4, 0): 1.0,
        (0, 4, 2, 0): 1.0,
    }
    _, _, dropped = drop_disconnected_arcs(z, {})
    assert dropped == {(0, 2, 3, 0), (0, 3, 4, 0), (0, 4, 2, 0)}


def test_values_away_from_zero_and_one_are_left_to_the_route_audit():
    z = {(0, 0, 1, 0): 1.0, (0, 1, 0, 0): 1.0, (0, 2, 3, 0): 0.4, (0, 3, 2, 0): 0.4}
    cleaned, _, dropped = drop_disconnected_arcs(z, {})
    assert dropped == set() and cleaned == z
    with pytest.raises(ValueError, match="não binários"):
        routes_from_z(cleaned, shape=(1, 4, 4, 1))


def test_cleaned_solution_passes_the_audit_and_costs_less():
    problem, variables = _with_cycle()
    assert not validate_variables(problem, expand_aggregated_vehicles(variables, 3))[
        "feasible"
    ]

    variables["Z"], variables["R"], _ = drop_disconnected_arcs(
        variables["Z"], variables["R"]
    )
    expanded = expand_aggregated_vehicles(variables, problem.v)
    assert validate_variables(problem, expanded)["feasible"]

    # Removing the cycle only lowers F5 (here by the 4 + 4 distance of the cycle).
    with_cycle = _with_cycle()[1]
    before = solution_components(problem, with_cycle)
    after = solution_components(problem, variables)
    np.testing.assert_allclose(after[:4], before[:4])
    assert before[4, 1] - after[4, 1] == pytest.approx(8.0)
    assert np.all(after <= before + 1e-12)


# --------------------------------------------------- extraction from the solver model


@pytest.fixture
def configured(monkeypatch):
    def apply(multiobjective):
        value = Config.normalize(
            {
                "solver": {
                    "multiobjective": multiobjective,
                    "method": "GUROBY",
                    "aggregate_vehicles": True,
                },
                "relaxed_solution": {"use": True},
            }
        )
        monkeypatch.setattr(Config, "_data", value)

    return apply


def _solver_with_incumbent(variables, model_objective):
    """The real aggregated model (not solved) reading a fake CPLEX incumbent."""
    problem, _ = _problem_and_aggregated_solution()
    data = _instance(customers=3, vehicles=3, periods=2, capacity=5, demand=0)
    data["d_pit"] = [[[2, 0], [1, 0], [4, 0]]]
    solver = MPPRP(data, "/tmp", Mock(), {"start": False})
    solver.solver(timeLimit=1, numThreads=1)

    values = {}
    for mapping, source in (
        (solver.model.X_p_t, lambda p, t: variables["X"][p, t]),
        (solver.model.Y_p_t, lambda p, t: variables["Y"][p, t]),
        (solver.model.I_p_i_t, lambda p, i, t: variables["I"][p, i, t]),
        (solver.model.Q_p_v_i_t, lambda p, v, i, t: variables["Q"][p, v, i, t]),
        (solver.model.R_p_v_i_k_t, lambda *key: variables["R"].get(key, 0.0)),
        (solver.model.Z_v_i_k_t, lambda *key: variables["Z"].get(key, 0.0)),
    ):
        for key, variable in mapping.items():
            values[variable] = source(*key)
    for variable in (*solver.positive.values(), solver.lambda_):
        values[variable] = 0.0

    attributes = dict(vars(solver.model))
    attributes.update(
        solution=SimpleNamespace(get_value=lambda variable: values[variable]),
        objective_value=model_objective,
        solve_details=SimpleNamespace(mip_relative_gap=0.02),
    )
    solver.model = SimpleNamespace(**attributes)
    solver.solCount, solver.time = 1, 1.0
    return problem, solver


def test_snapshot_drops_the_cycle_before_the_route_audit(configured):
    configured(multiobjective=False)
    problem, variables = _with_cycle()
    _, solver = _solver_with_incumbent(variables, model_objective=0.0)

    snapshot = solver.get_snapshot()

    assert solver.disconnected_arcs_removed == 2
    assert validate_variables(problem, snapshot)["feasible"]
    assert routes_from_z(snapshot["Z"], shape=(3, 4, 4, 2))[1] == [[], [], []]
    assert not any(key[:3] == (0, 1, 2) and key[3] == 1 for key in snapshot["Z"])


def test_a_solution_without_cycles_is_extracted_as_before(configured):
    configured(multiobjective=False)
    problem, variables = _problem_and_aggregated_solution()
    _, solver = _solver_with_incumbent(variables, model_objective=0.0)

    snapshot = solver.get_snapshot()

    assert solver.disconnected_arcs_removed == 0
    assert validate_variables(problem, snapshot)["feasible"]


def test_results_report_the_objective_of_the_cleaned_solution(configured):
    configured(multiobjective=False)
    problem, variables = _with_cycle()
    with_cycle_cost = float(solution_components(problem, variables).sum())
    _, solver = _solver_with_incumbent(variables, model_objective=with_cycle_cost)

    result = solver.getResults()
    cleaned = solver.get_variables()

    assert solver.disconnected_arcs_removed == 2
    assert result[7] == pytest.approx(with_cycle_cost - 8.0)
    assert result[7] == pytest.approx(
        evaluate_configured_objective(problem, cleaned, Config.snapshot())
    )
    assert result[11] == 1 and result[8] == 0.02
    assert validate_variables(problem, cleaned)["feasible"]


def test_goal_deviations_follow_the_cleaned_solution(configured):
    configured(multiobjective=True)
    problem, variables = _with_cycle()
    _, solver = _solver_with_incumbent(variables, model_objective=123.0)

    result = solver.getResults()
    cleaned = solver.get_variables()

    positive, _, limit = goal_start_values(problem, cleaned)
    np.testing.assert_allclose(result[6], positive.T)
    assert result[10] == pytest.approx(limit)
    assert result[7] == pytest.approx(
        evaluate_configured_objective(problem, cleaned, Config.snapshot())
    )
    assert result[7] != 123.0


# --------------------------------------------------------- the model admits cycles


def _model_with_cycle(monkeypatch, aggregate, multiobjective, load, idle=(1, 2)):
    data = _instance(customers=4, vehicles=2, periods=2, capacity=7, demand=3)
    data["I_pi0"] = [[0] + [20] * 4]  # period 0 needs no delivery
    value = Config.normalize(
        {
            "solver": {
                "multiobjective": multiobjective,
                "method": "GUROBY",
                "aggregate_vehicles": aggregate,
            },
            "relaxed_solution": {"use": True},
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    solver = MPPRP(data, "/tmp", Mock(), {"start": False})
    solver.solver(timeLimit=30, numThreads=1)
    model = solver.model
    model.add_constraint(model.Z_v_i_k_t[0, 1, 2, 0] == 1)
    model.add_constraint(model.Z_v_i_k_t[0, 2, 1, 0] == 1)
    for p in range(solver.p):
        model.add_constraint(model.R_p_v_i_k_t[p, 0, 1, 2, 0] == load)
        model.add_constraint(model.R_p_v_i_k_t[p, 0, 2, 1, 0] == load)
        for i in idle:
            model.add_constraint(model.Q_p_v_i_t[p, 0, i, 0] == 0)
    model.set_time_limit(30)
    return solver


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
@pytest.mark.parametrize("aggregate", [True, False])
@pytest.mark.parametrize("multiobjective", [False, True])
@pytest.mark.parametrize("load", [0.0, 3.0])
def test_model_admits_a_cycle_without_delivery_detached_from_the_plant(
    monkeypatch, aggregate, multiobjective, load
):
    solver = _model_with_cycle(monkeypatch, aggregate, multiobjective, load)
    try:
        assert solver.model.solve() is not None
    finally:
        solver.terminate()


@pytest.mark.cplex_integration
@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX não está instalado")
def test_model_does_not_admit_a_detached_cycle_with_delivery(monkeypatch):
    solver = _model_with_cycle(monkeypatch, True, False, 3.0, idle=(2,))
    try:
        model = solver.model
        # Customer 1 receives product 0 on the detached cycle: EQ_7 summed over
        # the cycle then needs an arc entering it, which EQ_12 forbids.
        model.add_constraint(model.Q_p_v_i_t[0, 0, 1, 0] >= 1)
        assert model.solve() is None
    finally:
        solver.terminate()
