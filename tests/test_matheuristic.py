import copy
import json
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from config import Config
from src.process.MatheuristicResults import write_artifacts
from src.solvers._routing_common import (
    canonicalize_vehicles,
    materialize_routes,
    ordered_arcs,
    routes_from_z,
    validate_variables,
)
from src.solvers._solver_common import (
    ProblemData,
    build_results_from_variables,
    empty_solution,
    evaluate_configured_objective,
    evaluate_solution_cost,
    goal_start_values,
)
from src.solvers.DeterministicMatheuristic import (
    DeterministicMatheuristic,
    result_variables,
)
from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)
from src.solvers.OrderedTsp import (
    _CACHE,
    build_tsp_model,
    initial_tour,
    solve_tsp,
    successor_cycles,
)
from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization


@pytest.fixture
def data():
    return {
        "file": "unit.dat",
        "num_products": 1,
        "num_customers": 2,
        "num_periods": 1,
        "num_vehicles": 2,
        "B": 10,
        "b_p": [1.5],
        "c_p": [2.5],
        "s_p": [3],
        "M": 10,
        "U_pi": [[20, 20, 20]],
        "I_pi0": [[0, 0, 0]],
        "h_pi": [[0, 0.5, 0.75]],
        "C": 5,
        "f": 7,
        "a_ik": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "d_pit": [[[0.5], [1.25]]],
        "weight": [0.2] * 5,
        "alpha": 0.2,
        "targets": [{f"f{j}_target": 10 for j in range(1, 6)}],
    }


@pytest.fixture
def configured(monkeypatch):
    value = Config.normalize(
        {
            "solver": {
                "multiobjective": False,
                "method": "GUROBY",
                "pso": {
                    "lot_sizing_bounds": {"enabled": False},
                    "execution_backend": "python",
                    "jit_warmup": False,
                    "swarm_size": 2,
                    "max_iterations": 0,
                    "use_as_mip_start": False,
                },
            }
        }
    )
    monkeypatch.setattr(Config, "_data", value)
    return value


@pytest.fixture
def variables(data):
    p = ProblemData.from_map(data)
    compact = empty_solution(p)
    compact["X"][0, 0] = 1.75
    compact["Y"][0, 0] = 1
    compact["Q"][0, 0, 1:, 0] = [0.5, 1.25]
    compact["route_plan"] = [[[1, 2], []]]
    return materialize_routes(p, compact)


def test_config_defaults_and_legacy_quantity_domain():
    value = Config.normalize(
        {"solver": {"pso": {"lot_sizing_bounds": {"integer_variables": True}}}}
    )
    assert value["solver"]["matheuristic"]["enabled"] is False
    assert value["solver"]["pso"]["time_limit"] is None
    assert value["solver"]["pso"]["lot_sizing_bounds"]["integer_variables"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"use_as": "wrong"},
        {"use_as": "pso_start"},
        {"restricted": {"time_limit": 0}},
    ],
)
def test_invalid_matheuristic_settings(mutation):
    with pytest.raises(ValueError):
        Config.normalize(
            {
                "solver": {
                    "method": "GUROBY",
                    "matheuristic": {"enabled": True, **mutation},
                }
            }
        )


@pytest.mark.parametrize(
    "extra",
    [{"postprocessing": {"build_target": True}}, {"relaxed_solution": {"use": True}}],
)
def test_incompatible_relaxation_flags(extra):
    with pytest.raises(ValueError, match="incompatível"):
        Config.normalize({"solver": {"matheuristic": {"enabled": True}}, **extra})


def test_fractional_quantities_cost_export_and_flow(data, variables, configured):
    p = ProblemData.from_map(data)
    assert validate_variables(p, variables)["feasible"]
    assert evaluate_solution_cost(p, variables) == pytest.approx(1.75 * 2.5 + 3 + 7 + 4)
    result = build_results_from_variables(p, variables, 18.375, 1)
    assert result[1][0][0] == 1.75
    assert result[5][0][0][0][1] == 0.5
    assert np.isnan(result[8])
    recovered = result_variables(p, result)
    for name in ("X", "Y", "I", "Q", "R", "Z"):
        np.testing.assert_allclose(variables[name], recovered[name])


def test_disconnected_cycle_is_rejected_even_with_zero_deliveries(data, variables):
    p = ProblemData.from_map(data)
    variables["Z"].fill(0)
    variables["Z"][0, 1, 2, 0] = variables["Z"][0, 2, 1, 0] = 1
    assert not validate_variables(p, variables)["feasible"]


def test_eta_and_order_validation(data, variables):
    p = ProblemData.from_map(data)
    assert validate_variables(p, variables, ordered_arcs([0, 1, 2, 0]))["feasible"]
    assert not validate_variables(p, variables, ordered_arcs([0, 2, 1, 0]))["feasible"]
    variables["eta"][0, 1, 0] = 0
    assert not validate_variables(p, variables)["feasible"]


def test_tsp_initial_cycles_and_model_without_solving(data):
    matrix = np.asarray(data["a_ik"])
    assert initial_tour(matrix) == [0, 1, 2, 0]
    assert successor_cycles([(0, 1), (1, 0), (2, 3), (3, 2)], 4) == [[0, 1], [2, 3]]
    model, u = build_tsp_model(matrix)
    try:
        assert len(u) == 6
        assert all(variable.is_binary() for variable in u.values())
        assert model.number_of_constraints == 6
    finally:
        model.end()


def test_tsp_cache_zero_customer_does_not_build_model(monkeypatch):
    _CACHE.clear()
    first = solve_tsp([[0]], 10, 1, Mock())
    second = solve_tsp([[0]], 10, 1, Mock())
    assert first["order"] == [0, 0]
    assert second["cache_hit"]
    second["u"][0, 0] = 9
    assert solve_tsp([[0]], 10, 1, Mock())["u"][0, 0] == 0


@pytest.mark.parametrize("coelho,hc1,bounds", list(product([False, True], repeat=3)))
def test_sparse_ordered_model_and_optional_features(
    data, configured, coelho, hc1, bounds
):
    data["strengthened_bounds"] = bounds
    model = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": False},
        routing_profile={"mode": "ordered", "order": [0, 1, 2, 0]},
        coelho_inequalities=coelho,
        symmetry_breaking_hc1=hc1,
    )
    try:
        model.createDecisionVariables()
        model.createRouteVisitDegrees()
        model.createVehicleLoadCapacityDelimited()
        model.crateObjectiveFunction()
        assert len(model.model.Z_v_i_k_t) == 10
        assert len(model.model.R_p_v_i_k_t) == 10
        assert all(
            variable.is_continuous() for variable in model.model.Z_v_i_k_t.values()
        )
        assert all(variable.is_binary() for variable in model.eta.values())
        assert model.model.Z_v_i_k_t[0, 2, 1, 0] == 0
        assert model.createCoelhoValidInequalities() == coelho
        assert model.createVehicleSymmetryBreaking() == hc1
        assert model.model.get_constraint_by_name("ETA_ASSIGN_1_0") is not None
    finally:
        model.terminate()


def test_improvement_fixed_quantities_and_no_empty_customer_visit(
    data, variables, configured
):
    fixed = {
        "X": variables["X"],
        "Y": variables["Y"],
        "delivery_totals": variables["Q"].sum(axis=1),
    }
    fixed["delivery_totals"][:, 2, :] = 0
    solver = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": False},
        routing_profile={"mode": "improvement"},
        fixed_plan=fixed,
        transport_objective=True,
    )
    try:
        solver.createDecisionVariables()
        solver.createRouteVisitDegrees()
        solver.crateObjectiveFunction()
        assert all(variable.is_binary() for variable in solver.model.Z_v_i_k_t.values())
        assert solver.model.get_constraint_by_name("FIX_X_0_0") is not None
        assert solver.model.get_constraint_by_name("FIX_Q_0_1_0") is not None
        assert solver.model.get_constraint_by_name("ETA_REQUIRED_2_0").rhs.constant == 0
        assert solver.model.objective_expr.get_coef(solver.model.X_p_t[0, 0]) == 0
    finally:
        solver.terminate()


def test_canonicalization_keeps_fractional_variables(data, variables):
    p = ProblemData.from_map(data)
    for name, axis in (("Q", 1), ("R", 1), ("Z", 0), ("eta", 0)):
        variables[name] = np.take(variables[name], [1, 0], axis=axis)
    result = canonicalize_vehicles(p, variables)
    assert routes_from_z(result["Z"]) == [[[1, 2], []]]
    assert result["Q"][0, 0, 2, 0] == 1.25
    assert validate_variables(p, result)["feasible"]


def test_seed_preserved_and_optional_deadline(data, variables, configured, monkeypatch):
    configured["solver"]["pso"]["time_limit"] = 1e-12
    configured["solver"]["pso"]["max_iterations"] = 100
    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=variables)
    solver._initialize_swarm()
    np.testing.assert_allclose(solver.population_state.Q[0], variables["Q"])
    assert solver.heuristic.solution_from_state(solver.population_state, 0)[
        "route_plan"
    ] == [[[1, 2], []]]
    solver.solver()
    assert solver.stopped_by_time_limit
    assert solver._current_iteration == 0
    assert solver.global_best_solution is not None


def test_seed_gene_intervals_obey_production_capacity_without_strengthening(
    data, variables, configured
):
    data["strengthened_bounds"] = False
    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=variables)
    solver._initialize_swarm()
    assert np.all(
        solver.heuristic.upper_bounds <= data["B"] / np.asarray(data["b_p"])[:, None]
    )
    assert np.all(solver.heuristic.upper_bounds >= variables["X"])
    np.testing.assert_allclose(solver.population_state.X[0], variables["X"])


def test_multiobjective_shared_score_and_mip_start(data, variables, configured):
    configured["solver"]["multiobjective"] = True
    p = ProblemData.from_map(data)
    positive, negative, limit = goal_start_values(p, variables)
    expected = data["alpha"] * limit + (1 - data["alpha"]) * np.sum(
        np.asarray(data["weight"])[:, None] * positive / 10
    )
    assert evaluate_configured_objective(p, variables, configured) == pytest.approx(
        expected
    )
    solver = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": True, "variables": variables},
        positive_only_deviations=False,
    )
    try:
        solver.createDecisionVariables()
        solver.crateObjectiveFunction()
        solver.startVariables()
        start = next(solver.model.iter_mip_starts())[0]
        assert start.get_value(solver.model.X_p_t[0, 0]) == 1.75
        assert start.get_value(solver.lambda_) == limit
        assert start.get_value(solver.negative[0, 0]) == negative[0, 0]
    finally:
        solver.terminate()


def test_fractional_python_numba_equivalence(data):
    reference = FeasibleParticleHeuristic(
        data, "/tmp", Mock(), execution_backend="python"
    )
    compiled = FeasibleParticleHeuristic(
        data, "/tmp", Mock(), execution_backend="numba", parallel_workers=1
    )
    particles = np.zeros((2, reference.particle_dim))
    a, b = reference.build_population_state(particles), compiled.build_population_state(
        particles
    )
    for key in (
        "X",
        "Y",
        "I",
        "Q",
        "costs",
        "route_nodes",
        "route_lengths",
        "feasible",
    ):
        np.testing.assert_allclose(getattr(a, key), getattr(b, key))
    assert a.X.dtype == b.X.dtype == np.float64


@pytest.mark.parametrize("debug", [False, True])
def test_complementary_parquets_and_sparse_metadata(tmp_path, data, variables, debug):
    saved = {
        "eta": variables["eta"],
        **({"X": variables["X"], "Q": variables["Q"]} if debug else {}),
    }
    payload = {
        "variables": [
            {"stage": "tsp", "variables": {"u": np.eye(3)}},
            {"stage": "restricted", "variables": saved},
        ],
        "stages": [
            {"stage": "restricted", "status": "mock", "gap_scope": "restricted"}
        ],
        "metadata": {"debug": debug, "sparse_zeroes": True},
    }
    run = write_artifacts(tmp_path, data, payload)
    df = pd.read_parquet(tmp_path / "matheuristic" / f"{run}.variables.parquet")
    assert {"u", "eta"}.issubset(set(df["var"]))
    assert ("X" in set(df["var"])) == debug
    assert not (tmp_path / "parquets").exists()
    meta = json.loads((tmp_path / "matheuristic" / f"{run}.metadata.json").read_text())
    assert meta["available_variables"]["restricted"]["eta"]["shape"] == [2, 3, 1]
    if debug:
        assert 0.5 in df.query("var == 'Q'")["value"].tolist()


@pytest.mark.parametrize("destination", ["final", "exact_start", "pso_start"])
@pytest.mark.parametrize("has_solution", [False, True])
def test_coordinator_destinations_and_no_incumbent(
    data, configured, variables, monkeypatch, destination, has_solution
):
    import src.solvers.DeterministicMatheuristic as module

    configured["solver"]["matheuristic"]["use_as"] = destination
    monkeypatch.setattr(
        module,
        "solve_tsp",
        lambda *args: {"order": [0, 1, 2, 0], "u": np.eye(3), "cache_hit": False},
    )
    wrapper = DeterministicMatheuristic(data, "/tmp", Mock())
    wrapper._constructive_start = Mock(return_value=None)
    wrapper._stage = Mock(return_value=variables if has_solution else None)
    result = build_results_from_variables(
        ProblemData.from_map(data),
        variables,
        18.375,
        1,
        solution_count=int(has_solution),
    )
    target = Mock()
    target.getResults.return_value = result
    exact = Mock(return_value=target)
    pso = Mock(return_value=target)
    monkeypatch.setattr(module, "MultProductProdctionRoutingProblem", exact)
    monkeypatch.setattr(
        "src.solvers.ParticleSwarmOptimization.ParticleSwarmOptimization", pso
    )
    wrapper.solver(timeLimit=4, numThreads=1)
    assert bool(wrapper.getResults()[11]) == has_solution
    if destination == "final":
        assert not exact.called and not pso.called
    elif destination == "exact_start":
        assert exact.call_args.args[3]["start"] == has_solution
    else:
        assert (pso.call_args.kwargs["initial_solution"] is not None) == has_solution


def test_stage3_transport_tradeoff_uses_configured_score(
    data, variables, configured, monkeypatch
):
    import src.solvers.DeterministicMatheuristic as module

    monkeypatch.setattr(
        module, "solve_tsp", lambda *args: {"order": [0, 1, 2, 0], "u": np.eye(3)}
    )
    wrapper = DeterministicMatheuristic(data, "/tmp", Mock())
    alternative = copy.deepcopy(variables)
    wrapper._constructive_start = Mock(return_value=None)
    wrapper._stage = Mock(side_effect=[variables, alternative])
    wrapper._score = lambda values: (0, 20) if values is variables else (1, 10)
    wrapper.solver()
    assert wrapper.selected_stage == "restricted"


def sparse(values):
    from src.solvers._routing_common import arc_items

    result = copy.deepcopy(values)
    for name in ("R", "Z"):
        result[name] = dict(arc_items(values[name]))
    return result


def test_sparse_snapshot_audit_cost_seed_and_export(data, variables, configured):
    p = ProblemData.from_map(data)
    snapshot = sparse(variables)
    assert validate_variables(p, snapshot, ordered_arcs([0, 1, 2, 0]))["feasible"]
    assert evaluate_solution_cost(p, snapshot) == evaluate_solution_cost(p, variables)
    result = build_results_from_variables(p, snapshot, 0, 1)
    np.testing.assert_allclose(result_variables(p, result)["R"], variables["R"])
    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=snapshot)
    solver._initialize_swarm()
    np.testing.assert_array_equal(solver.population_state.Q[0], variables["Q"])
    model = MPPRP(data, "/tmp", Mock(), {"start": True, "variables": snapshot})
    try:
        model.createDecisionVariables()
        model.startVariables()
        start = next(model.model.iter_mip_starts())[0]
        assert start.get_value(model.model.Q_p_v_i_t[0, 0, 2, 0]) == 1.25
        assert start.get_value(model.model.R_p_v_i_k_t[0, 0, 0, 1, 0]) == 1.75
    finally:
        model.terminate()


def test_tsp_timeout_callback_and_cache_limit(monkeypatch):
    import src.solvers.OrderedTsp as module
    from src.solvers.OrderedTsp import subtour_rows

    _CACHE.clear()
    indices = {
        (i, j): r
        for r, (i, j) in enumerate((i, j) for i in range(4) for j in range(4) if i != j)
    }
    assert subtour_rows([(0, 1), (1, 0), (2, 3), (3, 2)], 4, indices) == [
        ([indices[0, 1], indices[1, 0]], 1),
        ([indices[2, 3], indices[3, 2]], 1),
    ]
    models = []

    def model_double(distance):
        count = len(distance)
        model = Mock()
        model.parameters = SimpleNamespace(threads=1, randomseed=0)
        model.solve_details = SimpleNamespace(
            status="time limit", best_bound=0, mip_relative_gap=1
        )
        model.number_of_variables, model.number_of_constraints = (
            count * (count - 1),
            2 * count,
        )
        model.solve.return_value = None
        u = {
            (i, j): SimpleNamespace(index=r)
            for r, (i, j) in enumerate(
                (i, j) for i in range(count) for j in range(count) if i != j
            )
        }
        models.append(model)
        return model, u

    monkeypatch.setattr(module, "build_tsp_model", model_double)
    matrix = [[0, 1], [1, 0]]
    first = solve_tsp(matrix, 0.25, 1, Mock())
    assert first["order"] == [0, 1, 0] and first["status"] == "time limit"
    models[0].set_time_limit.assert_called_once_with(0.25)
    models[0].register_callback.assert_called_once()
    models[0].end.assert_called_once()
    assert solve_tsp(matrix, 0.25, 1, Mock())["cache_hit"]
    assert len(models) == 1
    for cost in range(2, 68):
        solve_tsp([[0, cost], [cost, 0]], 0.25, 1, Mock())
    assert len(_CACHE) == 64
    assert not solve_tsp(matrix, 0.25, 1, Mock())["cache_hit"]
    assert not solve_tsp(matrix, 0.5, 1, Mock())["cache_hit"]


def test_rounded_callback_omitted_arcs_and_eta_hc1(data, configured):
    configured["solver"]["rounded_capacity_inequalities"]["enabled"] = True
    solver = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": False},
        routing_profile={"mode": "ordered", "order": [0, 1, 2, 0]},
        symmetry_breaking_hc1=True,
    )
    try:
        solver.createDecisionVariables()
        names = [variable.name for variable in solver.model.iter_variables()]
        fake_cplex = Mock()
        fake_cplex.variables.get_names.return_value = names
        solver.model.get_cplex = Mock(return_value=fake_cplex)
        assert solver._prepare_rounded_capacity_callback_data()
        assert len(solver._rounded_capacity_z_indices) == len(solver.model.Z_v_i_k_t)
        assert all(
            (i, k) != (2, 1) for t, v, i, k, index in solver._rounded_capacity_z_indices
        )
        solver.createVehicleSymmetryBreaking()
        constraint = solver.model.get_constraint_by_name("SB_VC_v_1_t_0")
        assert constraint.rhs is solver.eta[0, 0, 0]
        # The callback row only contains instantiated crossing arcs.
        from src.solvers.RoundedCapacitySeparation import RoundedCapacityCut

        cut = RoundedCapacityCut(customers=(0, 1), tau=0, rhs=1, violation=1)
        indices, values = solver._rounded_capacity_build_row(cut)
        assert len(indices) == 8 and values == [1] * 8
    finally:
        solver.terminate()


@pytest.mark.parametrize("candidate_kind", ["none", "invalid", "changed_plan", "valid"])
def test_stage_snapshot_validation_cleanup_and_debug(
    data, configured, variables, monkeypatch, candidate_kind
):
    import src.solvers.DeterministicMatheuristic as module

    configured["solver"]["matheuristic"]["debug"]["enabled"] = True
    wrapper = DeterministicMatheuristic(data, "/tmp", Mock())
    wrapper.order = [0, 1, 2, 0]
    candidate = sparse(variables)
    fixed = {
        "X": variables["X"],
        "Y": variables["Y"],
        "delivery_totals": variables["Q"].sum(axis=1),
    }
    if candidate_kind == "invalid":
        candidate["Z"] = {(0, 1, 2, 0): 1, (0, 2, 1, 0): 1}
    elif candidate_kind == "changed_plan":
        candidate["X"] = candidate["X"] + 0.1
        candidate["I"][:, 0, :] += 0.1
    stage = Mock()
    stage.solCount = int(candidate_kind != "none")
    stage.get_snapshot.return_value = candidate
    stage.get_auxiliary_variables.return_value = {"lambda": np.array(0.0)}
    stage.get_telemetry.return_value = {"rounded_capacity": {}}
    stage.applied_features = {"hc1": False}
    stage.time = 0.01
    stage.model.solve_details = SimpleNamespace(
        status="mock", best_bound=0, mip_relative_gap=0.5
    )
    stage.model.objective_value = 11
    stage.model.number_of_variables, stage.model.number_of_constraints = 10, 20
    monkeypatch.setattr(
        module, "MultProductProdctionRoutingProblem", Mock(return_value=stage)
    )
    selected = wrapper._stage(
        "route_improvement", variables, 1, 1, {"mode": "improvement"}, fixed
    )
    assert (selected is not None) == (candidate_kind == "valid")
    stage.terminate.assert_called_once()
    assert bool(wrapper.artifacts) == (candidate_kind != "none")
    assert wrapper.stages[-1]["has_incumbent"] == (candidate_kind != "none")
    assert wrapper.stages[-1]["has_solution"] == (candidate_kind == "valid")
    if selected is not None:
        assert isinstance(selected["Z"], dict)
        assert "eta" in wrapper.artifacts[-1]["variables"]
        assert isinstance(wrapper.artifacts[-1]["variables"]["R"], dict)
        assert all(
            isinstance(value, (float, np.floating))
            for value in wrapper.artifacts[-1]["variables"]["R"].values()
        )
    assert wrapper.stages[-1]["gap_scope"] == "route_improvement"


def test_empty_period_and_multi_product_fractional_routes(data, configured):
    data.update(
        num_products=2,
        num_periods=2,
        b_p=[1, 1.5],
        c_p=[2.5, 3.5],
        s_p=[3, 4],
        U_pi=[[20] * 3] * 2,
        I_pi0=[[0] * 3] * 2,
        h_pi=[[0.5] * 3] * 2,
        d_pit=[[[0.5, 0], [1.25, 0]], [[0.25, 0], [0.75, 0]]],
    )
    p = ProblemData.from_map(data)
    compact = empty_solution(p)
    compact["Q"][:, 0, 1:, 0] = [[0.5, 1.25], [0.25, 0.75]]
    compact["X"][:, 0] = [1.75, 1]
    compact["Y"][:, 0] = 1
    compact["route_plan"] = [[[1, 2], []], [[], []]]
    values = materialize_routes(p, compact)
    assert validate_variables(p, values)["feasible"]
    assert values["R"][1, 0, 0, 1, 0] == 1
    fixed = {
        "X": values["X"],
        "Y": values["Y"],
        "delivery_totals": values["Q"].sum(axis=1),
    }
    solver = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": False},
        routing_profile={"mode": "improvement"},
        fixed_plan=fixed,
    )
    try:
        solver.createDecisionVariables()
        solver.createRouteVisitDegrees()
        assert not solver.arcs_by_period[1]
        assert solver.model.get_constraint_by_name("ETA_REQUIRED_1_1").rhs.constant == 0
    finally:
        solver.terminate()


def test_multiobjective_prefers_restricted_despite_lower_transport(
    data, variables, configured, monkeypatch
):
    import src.solvers.DeterministicMatheuristic as module

    configured["solver"]["multiobjective"] = True
    data.update(weight=[0, 0, 0, 1, 0], a_ik=[[0, 1, 1], [1, 0, 10], [1, 10, 0]])
    data["targets"][0]["f4_target"] = 7
    p = ProblemData.from_map(data)
    alternative = copy.deepcopy(variables)
    alternative["Q"][0, 1, 2, 0], alternative["Q"][0, 0, 2, 0] = 1.25, 0
    alternative["route_plan"] = [[[1], [2]]]
    alternative = materialize_routes(p, alternative)
    assert evaluate_solution_cost(p, alternative) < evaluate_solution_cost(p, variables)
    assert evaluate_configured_objective(
        p, alternative, configured
    ) > evaluate_configured_objective(p, variables, configured)
    wrapper = DeterministicMatheuristic(data, "/tmp", Mock())
    monkeypatch.setattr(
        module, "solve_tsp", lambda *args: {"order": [0, 1, 2, 0], "u": np.eye(3)}
    )
    wrapper._constructive_start = Mock(return_value=None)
    wrapper._stage = Mock(side_effect=[variables, alternative])
    wrapper.solver()
    assert wrapper.selected_stage == "restricted"


def test_pso_deadline_with_controlled_clock(data, variables, configured, monkeypatch):
    import src.solvers.ParticleSwarmOptimization as module

    configured["solver"]["pso"].update(time_limit=3, max_iterations=100)
    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=variables)
    clock = {"now": 0.0}
    initialize = solver._initialize_swarm

    def initialize_and_advance():
        initialize()
        clock["now"] = 4.0

    monkeypatch.setattr(module.time, "perf_counter", lambda: clock["now"])
    solver._initialize_swarm = initialize_and_advance
    solver._update_positions = Mock()
    solver.solver()
    solver._update_positions.assert_not_called()
    assert solver.stopped_by_time_limit and solver.global_best_solution is not None
    assert solver.get_telemetry()["timed_out"]


def test_dispatch_disabled_preserves_existing_solver(data, configured, monkeypatch):
    import src.process.InstanceProcess as module

    exact, pso = Mock(), Mock()
    monkeypatch.setattr(module, "MPPRP", exact)
    monkeypatch.setattr(module, "PSOSolver", pso)
    instance = module.InstanceProcess("unit.dat", "/tmp", log=Mock(), solver="GUROBY")
    assert instance.solverInstancie(data) is exact.return_value
    instance.solver = "PSO"
    assert instance.solverInstancie(data) is pso.return_value
    configured["solver"]["matheuristic"]["enabled"] = True
    assert isinstance(instance.solverInstancie(data), DeterministicMatheuristic)


def test_quantity_lot_sizing_domains_even_legacy_true(data):
    from src.solvers.LotSizingRelaxation import LotSizingRelaxation

    solver = LotSizingRelaxation(data, Mock(), integer_variables=True)
    model, x, inventory, q = solver._build_model()
    try:
        assert all(
            var.is_continuous()
            for var in (*x.values(), *inventory.values(), *q.values())
        )
    finally:
        model.end()


@pytest.mark.parametrize("exact_quality", ["missing", "worse", "invalid", "same"])
def test_pso_keeps_best_after_optional_exact(
    data, variables, configured, exact_quality
):
    from src.solvers._routing_common import compact_solution

    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=variables)
    solver.global_best_solution = compact_solution(solver.problem, variables)
    solver.global_best_cost = evaluate_solution_cost(solver.problem, variables)
    solver.pso_config.update(
        use_as_mip_start=True, return_heuristic_result_without_cplex=False
    )
    candidate = copy.deepcopy(variables)
    if exact_quality == "worse":
        candidate["X"] += 0.5
        candidate["I"][:, 0, :] += 0.5
    elif exact_quality == "invalid":
        candidate["Z"].fill(0)
    result = build_results_from_variables(
        solver.problem, candidate, 0, 1, solution_count=int(exact_quality != "missing")
    )
    solver.solverGurobi = Mock()
    solver.solverGurobi.getResults.return_value = result
    solver.solverGurobi.get_telemetry.return_value = {
        "status": "time limit exceeded",
        "timed_out": True,
        "objective": result[7],
        "best_bound": 2.0,
        "root_bound": 1.0,
        "relative_gap": 0.1,
        "gap_target_seconds": 1.0,
        "mip_events": [],
    }
    chosen = result_variables(solver.problem, solver.getResults())
    assert validate_variables(solver.problem, chosen)["feasible"]
    np.testing.assert_allclose(chosen["X"], variables["X"])
    telemetry = solver.get_telemetry()
    if exact_quality == "same":
        assert telemetry["selected_solution"] == "exact"
        assert telemetry["relative_gap"] == 0.1
    else:
        assert telemetry["selected_solution"] == "pso"
        assert telemetry["objective"] == solver.global_best_cost
        assert telemetry["relative_gap"] is None
        assert telemetry["best_bound"] is None and telemetry["root_bound"] is None
        assert telemetry["gap_target_seconds"] is None
    assert telemetry["timed_out"]


@pytest.mark.parametrize("multi", [False, True])
def test_multiple_fractional_periods_python_numba_and_goals(data, multi):
    data.update(
        num_products=2,
        num_periods=2,
        b_p=[1, 1.5],
        c_p=[2.5, 3.5],
        s_p=[3, 4],
        U_pi=[[20] * 3] * 2,
        I_pi0=[[0] * 3] * 2,
        h_pi=[[0.5] * 3] * 2,
        d_pit=[[[0.5, 0.25], [1.25, 0.5]], [[0.25, 0.75], [0.75, 0.25]]],
    )
    data["targets"] *= 2
    cfg = {"solver": {"multiobjective": multi}}
    a = FeasibleParticleHeuristic(
        data, "/tmp", Mock(), execution_backend="python", objective_config=cfg
    )
    b = FeasibleParticleHeuristic(
        data,
        "/tmp",
        Mock(),
        execution_backend="numba",
        parallel_workers=1,
        objective_config=cfg,
    )
    genes = np.random.default_rng(20).normal(size=(3, a.particle_dim))
    python, compiled = a.build_population_state(genes), b.build_population_state(genes)
    for name in (
        "X",
        "Y",
        "I",
        "Q",
        "costs",
        "feasible",
        "route_nodes",
        "route_lengths",
    ):
        np.testing.assert_allclose(getattr(python, name), getattr(compiled, name))
    for index in np.flatnonzero(compiled.feasible):
        values = b.solution_from_state(compiled, index)
        assert compiled.costs[index] == pytest.approx(
            evaluate_configured_objective(b.problem, values, cfg)
        )


def test_fractional_final_files_keep_schema_and_shared_cost(tmp_path, data, variables):
    from src.process.ProcessResults import getResults

    p = ProblemData.from_map(data)
    result = build_results_from_variables(
        p, variables, evaluate_solution_cost(p, variables), 1
    )
    getResults(data, tmp_path, *result)
    final = pd.read_parquet(next((tmp_path / "parquets").glob("*.parquet")))
    assert not (tmp_path / "matheuristic").exists()
    assert 0.5 in final.loc[final["var"].eq("Q"), "value"].tolist()
    assert 1.75 in final.loc[final["var"].eq("X"), "value"].tolist()
    excel = pd.read_excel(next(tmp_path.glob("*.xlsx")))
    assert excel.loc[0, ["f1", "f2", "f3", "f4", "f5"]].sum() == pytest.approx(
        result[7]
    )
    run = write_artifacts(
        tmp_path,
        data,
        {
            "variables": [],
            "stages": [{"stage": "tsp", "status": "mock"}],
            "metadata": {},
        },
    )
    meta = json.loads((tmp_path / "matheuristic" / f"{run}.metadata.json").read_text())
    assert (
        meta["final_output"]["mtime_ns"]
        == next((tmp_path / "parquets").glob("*.parquet")).stat().st_mtime_ns
    )


def test_notebook_is_unexecuted_and_syntactically_valid():
    import ast
    from pathlib import Path

    notebook = json.loads(
        (Path(__file__).parents[1] / "analysis/matheuristic_debug.ipynb").read_text()
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), False, -1])
def test_deadlines_must_be_finite_positive(bad):
    with pytest.raises(ValueError):
        Config.normalize({"solver": {"matheuristic": {"tsp": {"time_limit": bad}}}})
    with pytest.raises(ValueError):
        Config.normalize({"solver": {"pso": {"time_limit": bad}}})


def test_relaxed_target_rebinds_variables_without_solving(
    data, configured, monkeypatch
):
    from docplex.mp.model import Model

    solver = MPPRP(data, "/tmp", Mock(), {"start": False})
    solver.createDecisionVariables()
    solver.crateObjectiveFunction()

    # Controlled values represent a fractional relaxation; no optimization occurs.
    def solve_double(model, *args, **kwargs):
        solution = model.new_solution(objective_value=10)
        solution.add_var_value(model.get_var_by_name("X_0_0"), 1.75)
        solution.add_var_value(model.get_var_by_name("Y_0_0"), 0.5)
        solution.add_var_value(model.get_var_by_name("Z_0_0_1_0"), 0.5)
        model._set_solution(solution)
        model._solve_details = SimpleNamespace(mip_relative_gap=float("nan"))
        return solution

    monkeypatch.setattr(Model, "solve", solve_double)
    try:
        solver.generteRelax(REPLACE_MODEL=True)
        solver.solCount = 1
        result = solver.getResults()
        assert result[0][0][0][0][1] == 0.5
        assert result[2][0][0] == 0.5
        assert result[1][0][0] == 1.75
        assert solver.model.X_p_t[0, 0].model is solver.model
        assert solver.lambda_.model is solver.model
    finally:
        solver.terminate()


def test_snapshot_direct_extraction_and_sparse_hc1(data, variables, configured):
    from docplex.mp.model import Model

    p = ProblemData.from_map(data)
    solver = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": False},
        routing_profile={"mode": "ordered", "order": [0, 1, 2, 0]},
    )
    try:
        solver.createDecisionVariables()
        solution = solver.model.new_solution()
        for name, mapping in (
            ("X", solver.model.X_p_t),
            ("Y", solver.model.Y_p_t),
            ("I", solver.model.I_p_i_t),
            ("Q", solver.model.Q_p_v_i_t),
            ("R", solver.model.R_p_v_i_k_t),
            ("Z", solver.model.Z_v_i_k_t),
            ("eta", solver.eta),
        ):
            for key, var in mapping.items():
                solution.add_var_value(var, variables[name][key])
        solver.model._set_solution(solution)
        snapshot = solver.get_snapshot()
        assert isinstance(snapshot["Z"], dict) and isinstance(snapshot["R"], dict)
        assert validate_variables(p, snapshot)["feasible"]
        dense = solver.get_variables()
        np.testing.assert_allclose(dense["R"], variables["R"])
        for name, axis in (("Q", 1), ("R", 1), ("Z", 0), ("eta", 0)):
            dense[name] = np.take(dense[name], [1, 0], axis=axis)
        canonical = canonicalize_vehicles(p, sparse(dense))
        assert validate_variables(p, canonical)["feasible"]
        assert canonical["Z"][0, 0, 1, 0] == 1
        assert canonical["R"][0, 0, 0, 1, 0] == 1.75
    finally:
        solver.terminate()


def test_sparse_callback_uses_vehicle_aggregation(data, configured, monkeypatch):
    import src.solvers.MultProductProdctionRoutingProblem as module

    configured["solver"]["rounded_capacity_inequalities"]["enabled"] = True
    solver = MPPRP(
        data,
        "/tmp",
        Mock(),
        {"start": False},
        routing_profile={"mode": "ordered", "order": [0, 1, 2, 0]},
    )
    try:
        solver.createDecisionVariables()
        fake = Mock()
        fake.variables.get_names.return_value = [
            var.name for var in solver.model.iter_variables()
        ]
        solver.model.get_cplex = Mock(return_value=fake)
        solver._prepare_rounded_capacity_callback_data()
        callback = Mock()
        callback.get_values.return_value = [0.5] * len(
            solver._rounded_capacity_z_indices
        )
        separate = Mock(return_value=[])
        monkeypatch.setattr(module, "separate_cumulative_cuts", separate)
        solver._run_rounded_capacity_separator(callback, 1)
        values = separate.call_args.args[0]
        assert values.shape == (1, 3, 3)
        assert values[0, 2, 1] == 0
        assert values[0, 0, 1] == 1  # Two vehicle values of .5.
        from src.solvers.RoundedCapacitySeparation import aggregate_route_arcs

        np.testing.assert_allclose(
            aggregate_route_arcs(values, 0),
            aggregate_route_arcs(values[:, None, :, :], 0),
        )
    finally:
        solver.terminate()


def test_auxiliary_and_numba_threads_respect_mpi_limit(data, configured, monkeypatch):
    from src.solvers.LotSizingRelaxation import LotSizingRelaxation

    configured["solver"]["threadsLimit"] = 2
    configured["solver"]["pso"]["parallel_workers"] = 8
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "16")
    solver = ParticleSwarmOptimization(data, "/tmp", Mock())
    assert solver.parallel_workers == 2
    relaxation = LotSizingRelaxation(data, Mock())
    model, *_ = relaxation._build_model()
    try:
        relaxation._apply_time_limit(model)
        assert model.parameters.threads.get() == 2
    finally:
        model.end()


def test_debug_writes_sparse_arc_maps_without_materialization(
    tmp_path, data, variables
):
    snapshot = sparse(variables)
    payload = {
        "variables": [{"stage": "restricted", "variables": snapshot}],
        "stages": [{"stage": "restricted", "status": "mock"}],
        "metadata": {"debug": True, "dimensions": {"p": 1, "v": 2, "i": 3, "t": 1}},
    }
    run = write_artifacts(tmp_path, data, payload)
    saved = pd.read_parquet(tmp_path / "matheuristic" / f"{run}.variables.parquet")
    r = saved.loc[saved["var"].eq("R")]
    assert len(r) == len(snapshot["R"])
    assert (r["value"] != 0).all() and 1.75 in r["value"].tolist()
    metadata = json.loads(
        (tmp_path / "matheuristic" / f"{run}.metadata.json").read_text()
    )
    assert metadata["available_variables"]["restricted"]["R"]["shape"] == [
        1,
        2,
        3,
        3,
        1,
    ]
    assert metadata["available_variables"]["restricted"]["R"]["sparse"]


def test_stage_summary_writes_mixed_stage_shapes(tmp_path, data):
    payload = {
        "variables": [],
        "metadata": {},
        "stages": [
            {"stage": "tsp", "gap_scope": "tsp", "status": "mock", "gap": None},
            {
                "stage": "restricted",
                "gap_scope": "restricted",
                "status": "mock",
                "features": {"coelho": True, "hc1": False},
                "rounded_capacity": {"cuts_added": 0},
                "costs": [[1.25], [2.0], [0.0], [7.0], [4.0]],
                "validation_errors": [],
                "gap": 0.5,
            },
            {"stage": "exact_start", "gap_scope": "unrestricted", "status": "mock"},
        ],
    }
    run = write_artifacts(tmp_path, data, payload)
    table = pd.read_parquet(tmp_path / "matheuristic" / f"{run}.stages.parquet")
    assert table["gap_scope"].tolist() == ["tsp", "restricted", "unrestricted"]
    assert table.iloc[1]["features"]["coelho"]
    np.testing.assert_allclose(
        np.stack(table.iloc[1]["costs"]), payload["stages"][1]["costs"]
    )
