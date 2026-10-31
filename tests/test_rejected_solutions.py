"""How a rejected MIP incumbent shows up in the results (production run of 2026-10-29).

In the experiments with the unrestricted MIP, ``validate_variables`` rejected the CPLEX
incumbent in 318 of 900 ``matheuristic_mip_polish`` tasks ("Rota desconectada da
planta") and the heuristic solution was kept. The tests below pin down what the audit
reports (every violation, not only the first routing defect) and what the pipeline
leaves behind: a warning in the log and a stage record with the reason, the MIP
objective and its gap.
"""

import copy
from unittest.mock import Mock

import numpy as np
import pytest
from test_matheuristic import configured, data, variables  # noqa: F401

from src.solvers._routing_common import validate_variables
from src.solvers._solver_common import (
    ProblemData,
    build_results_from_variables,
    evaluate_solution_cost,
)
from src.solvers.DeterministicMatheuristic import result_variables
from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization

CYCLE_MESSAGE = "Rota desconectada da planta"


def _add_detached_cycle(candidate):
    # Vehicle 1 runs 1 -> 2 -> 1 without the plant and without load.
    candidate["Z"][1, 1, 2, 0] = 1
    candidate["Z"][1, 2, 1, 0] = 1
    return candidate


def test_detached_cycle_is_reported_with_the_production_message(data, variables):
    problem = ProblemData.from_map(data)
    report = validate_variables(problem, _add_detached_cycle(copy.deepcopy(variables)))
    assert report == {"feasible": False, "violations": [CYCLE_MESSAGE]}


def test_route_failure_does_not_hide_other_violations(data, variables):
    problem = ProblemData.from_map(data)
    candidate = _add_detached_cycle(copy.deepcopy(variables))
    candidate["I"][0, 1, 0] += 1.0  # customer stock no longer balances
    violations = validate_variables(problem, candidate)["violations"]
    assert CYCLE_MESSAGE in violations
    assert "Balanço de estoque nos clientes" in violations


def _solver_with_exact_result(data, variables, candidate):
    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=variables)
    from src.solvers._routing_common import compact_solution

    solver.global_best_solution = compact_solution(solver.problem, variables)
    solver.global_best_cost = evaluate_solution_cost(solver.problem, variables)
    solver.pso_config.update(
        use_as_mip_start=True, return_heuristic_result_without_cplex=False
    )
    result = build_results_from_variables(solver.problem, candidate, 0, 1, solution_count=1)
    solver.solverGurobi = Mock()
    solver.solverGurobi.getResults.return_value = result
    solver.solverGurobi.time = 1.0
    return solver


def test_pso_keeps_the_heuristic_when_the_exact_incumbent_has_a_cycle(
    data, variables, configured  # noqa: F811
):
    candidate = _add_detached_cycle(copy.deepcopy(variables))
    solver = _solver_with_exact_result(data, variables, candidate)

    chosen = result_variables(solver.problem, solver.getResults())

    np.testing.assert_allclose(chosen["X"], variables["X"])
    assert validate_variables(solver.problem, chosen)["feasible"]


def test_pso_logs_why_the_exact_incumbent_was_discarded(
    data, variables, configured  # noqa: F811
):
    candidate = _add_detached_cycle(copy.deepcopy(variables))
    solver = _solver_with_exact_result(data, variables, candidate)

    solver.getResults()

    messages = " ".join(str(call) for call in solver.log.warning.call_args_list)
    assert CYCLE_MESSAGE in messages


@pytest.mark.parametrize(
    "excess, rejected", [(5e-7, False), (2e-6, True)], ids=["inside", "outside"]
)
def test_capacity_audit_uses_an_absolute_tolerance_of_one_micro_unit(
    data, variables, excess, rejected
):
    # Hypothesis for the six matheuristic tasks lost to 'Produção/setup/capacidade':
    # with B ~ 2.6e4 in production, 1e-6 is a relative tolerance of ~4e-11, tighter
    # than CPLEX's own feasibility tolerance. The test only fixes the current rule.
    problem = ProblemData.from_map(data)
    candidate = copy.deepcopy(variables)
    # Setup time/capacity use b_p: production hits the capacity B plus the excess.
    candidate["X"][0, 0] = (problem.B + excess) / np.asarray(problem.b_p)[0]
    violations = validate_variables(problem, candidate)["violations"]
    assert ("Produção/setup/capacidade" in violations) is rejected


def test_pso_records_the_discarded_exact_incumbent(
    data, variables, configured  # noqa: F811
):
    candidate = _add_detached_cycle(copy.deepcopy(variables))
    solver = _solver_with_exact_result(data, variables, candidate)
    assert solver.get_artifacts() is None  # nothing before the exact result is read

    solver.getResults()
    artifacts = solver.get_artifacts()

    (record,) = artifacts["stages"]
    assert record["stage"] == "exact" and record["gap_scope"] == "unrestricted"
    assert record["has_solution"] and not record["selected"]
    assert CYCLE_MESSAGE in record["validation_errors"]
    assert record["objective"] == 0 and record["gap"] is not None
    assert artifacts["metadata"]["selected_stage"] == "pso"
    assert artifacts["variables"] == []


def test_pso_without_exact_stage_has_no_artifacts(
    data, variables, configured  # noqa: F811
):
    solver = ParticleSwarmOptimization(data, "/tmp", Mock(), initial_solution=variables)
    solver.pso_config["return_heuristic_result_without_cplex"] = True
    solver.getResults()
    assert solver.get_artifacts() is None


def test_exact_incumbent_that_is_worse_is_logged_without_a_warning(
    data, variables, configured  # noqa: F811
):
    candidate = copy.deepcopy(variables)
    candidate["X"] += 0.5
    candidate["I"][:, 0, :] += 0.5
    solver = _solver_with_exact_result(data, variables, candidate)

    solver.getResults()

    solver.log.warning.assert_not_called()
    assert "não melhora" in str(solver.log.info.call_args_list)
    assert solver.get_artifacts()["stages"][0]["validation_errors"] == []


def test_pso_artifacts_are_written_as_a_stage_record(
    data, variables, configured, tmp_path  # noqa: F811
):
    import pandas as pd

    from src.process.MatheuristicResults import write_artifacts

    candidate = _add_detached_cycle(copy.deepcopy(variables))
    solver = _solver_with_exact_result(data, variables, candidate)
    solver.getResults()

    run = write_artifacts(tmp_path, data, solver.get_artifacts())

    stages = pd.read_parquet(tmp_path / "matheuristic" / f"{run}.stages.parquet")
    assert stages.stage.tolist() == ["exact"]
    assert stages.selected.tolist() == [False]
    assert stages.validation_errors.iloc[0].tolist() == [CYCLE_MESSAGE]
