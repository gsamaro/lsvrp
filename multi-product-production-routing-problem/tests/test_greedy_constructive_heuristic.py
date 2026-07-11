import sys
import types
import unittest
import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

docplex_module = types.ModuleType("docplex")
docplex_mp_module = types.ModuleType("docplex.mp")
docplex_mp_model_module = types.ModuleType("docplex.mp.model")
docplex_mp_relax_module = types.ModuleType("docplex.mp.relax_linear")


class FakeWarmStart:
    def __init__(self):
        self.values = []

    def add_var_value(self, var, value):
        self.values.append(value)


class FakeModel:
    def __init__(self, name=None):
        self.name = name
        self.X_p_t = {(0, 0): object()}
        self.Y_p_t = {(0, 0): object()}
        self.I_p_i_t = {(0, 0, 0): object()}
        self.Z_v_i_k_t = {(0, 0, 0, 0): object()}
        self.R_p_v_i_k_t = {(0, 0, 0, 0, 0): object()}
        self.Q_p_v_i_t = {(0, 0, 0, 0): object()}
        self.added = None

    def new_solution(self):
        return FakeWarmStart()

    def add_mip_start(self, warm_start):
        self.added = warm_start


docplex_mp_model_module.Model = FakeModel
docplex_mp_relax_module.LinearRelaxer = object

sys.modules.setdefault("docplex", docplex_module)
sys.modules.setdefault("docplex.mp", docplex_mp_module)
sys.modules.setdefault("docplex.mp.model", docplex_mp_model_module)
sys.modules.setdefault("docplex.mp.relax_linear", docplex_mp_relax_module)

solver_module = types.ModuleType("src.solvers.MultProductProdctionRoutingProblem")
solver_module.MultProductProdctionRoutingProblem = object
sys.modules.setdefault("src.solvers.MultProductProdctionRoutingProblem", solver_module)

from src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic import (  # noqa: E402
    MultProductProdctionRoutingProblemGreedyConstructiveHeuristic,
)
import src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic as heuristic_module  # noqa: E402


def _load_mpprp_module():
    module_name = "mpprp_real_for_tests"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "src/solvers/MultProductProdctionRoutingProblem.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class GreedyConstructiveHeuristicTestCase(unittest.TestCase):
    def _minimal_data(self, demand=5, production_capacity=1, vehicle_capacity=10):
        return {
            "num_products": 1,
            "num_customers": 1,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": production_capacity,
            "b_p": [1],
            "c_p": [1],
            "s_p": [1],
            "M": 100,
            "U_pi": [[10, 10]],
            "I_pi0": [[0, 0]],
            "h_pi": [[0, 0]],
            "C": vehicle_capacity,
            "f": 1,
            "a_ik": [[0, 1], [1, 0]],
            "d_pit": [[[demand]]],
        }

    def test_solver_keeps_inventory_non_negative_on_minimal_instance(self):
        data = self._minimal_data()

        heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            rng=np.random.default_rng(123),
        )

        heuristic.solver()

        self.assertGreaterEqual(heuristic.variables["I"][0][1][0], 0)
        self.assertEqual(heuristic.variables["X"][0][0], 1)
        self.assertLessEqual(sum(heuristic.variables["X"][0]), data["B"])

    def test_validate_feasibility_accepts_balanced_solution(self):
        data = self._minimal_data(demand=5, production_capacity=10, vehicle_capacity=10)
        heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            rng=np.random.default_rng(123),
        )

        X = np.zeros((1, 1), dtype=int)
        Y = np.zeros((1, 1), dtype=int)
        I = np.zeros((1, 2, 1), dtype=int)
        Q = np.zeros((1, 1, 2, 1), dtype=int)
        R = np.zeros((1, 1, 2, 2, 1), dtype=int)
        Z = np.zeros((1, 2, 2, 1), dtype=int)

        X[0, 0] = 5
        Y[0, 0] = 1
        Q[0, 0, 1, 0] = 5
        R[0, 0, 0, 1, 0] = 5
        Z[0, 0, 1, 0] = 1
        Z[0, 1, 0, 0] = 1
        heuristic.variables = {"X": X, "Y": Y, "I": I, "Q": Q, "R": R, "Z": Z}

        report = heuristic.validateFeasibility()

        self.assertTrue(report["feasible"], report["violations"])

    def test_validate_feasibility_rejects_unbalanced_heuristic_solution(self):
        data = self._minimal_data()
        heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            rng=np.random.default_rng(123),
        )

        heuristic.solver()

        self.assertFalse(heuristic.feasibility_report["feasible"])
        self.assertTrue(
            any("balanco do cliente" in item for item in heuristic.feasibility_report["violations"])
        )

    def test_solver_does_not_pass_infeasible_heuristic_as_warm_start(self):
        data = self._minimal_data()
        starts = []

        class FakeSolver:
            def __init__(self, map, dir, log, start):
                starts.append(start)

            def solver(self, timeLimit=None, numThreads=None):
                pass

            def getResults(self):
                return None

        original_mpprp = heuristic_module.MPPRP
        heuristic_module.MPPRP = FakeSolver
        try:
            heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
                map=data,
                dir="/tmp",
                log=DummyLogger(),
                rng=np.random.default_rng(123),
            )
            heuristic.setMitStart(True)
            heuristic.solver()
        finally:
            heuristic_module.MPPRP = original_mpprp

        self.assertEqual(starts, [{"start": False}])

    def test_repair_solution_rebuilds_feasible_minimal_solution(self):
        data = self._minimal_data(demand=5, production_capacity=10, vehicle_capacity=10)
        heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            rng=np.random.default_rng(123),
        )
        heuristic.solution = {"routes": []}
        heuristic.variables = {}

        repaired = heuristic.repairSolution()
        report = heuristic.validateFeasibility()

        self.assertTrue(repaired)
        self.assertTrue(report["feasible"], report["violations"])
        self.assertEqual(heuristic.variables["X"][0][0], 5)
        self.assertEqual(heuristic.variables["Y"][0][0], 1)
        self.assertEqual(heuristic.variables["Q"][0][0][1][0], 5)
        self.assertEqual(heuristic.variables["R"][0][0][0][1][0], 5)
        self.assertEqual(heuristic.variables["Z"][0][0][1][0], 1)
        self.assertEqual(heuristic.variables["Z"][0][1][0][0], 1)

    def test_solver_passes_repaired_solution_as_warm_start(self):
        data = self._minimal_data(demand=5, production_capacity=10, vehicle_capacity=10)
        starts = []

        class FakeSolver:
            def __init__(self, map, dir, log, start):
                starts.append(start)

            def solver(self, timeLimit=None, numThreads=None):
                pass

            def getResults(self):
                return None

        original_mpprp = heuristic_module.MPPRP
        heuristic_module.MPPRP = FakeSolver
        try:
            heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
                map=data,
                dir="/tmp",
                log=DummyLogger(),
                rng=np.random.default_rng(123),
            )
            heuristic.setMitStart(True)
            heuristic.solver()
        finally:
            heuristic_module.MPPRP = original_mpprp

        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0]["start"])
        repaired_report = heuristic.validateFeasibility()
        self.assertTrue(repaired_report["feasible"], repaired_report["violations"])

    def test_repair_solution_rejects_vehicle_capacity_infeasibility(self):
        data = self._minimal_data(demand=5, production_capacity=10, vehicle_capacity=4)
        heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            rng=np.random.default_rng(123),
        )
        heuristic.solution = {"routes": []}
        heuristic.variables = {}

        repaired = heuristic.repairSolution()
        report = heuristic.validateFeasibility()

        self.assertFalse(repaired)
        self.assertFalse(report["feasible"])

    def test_construir_solucao_preserves_depot_and_customer_demand_order(self):
        data = {
            "num_products": 1,
            "num_customers": 2,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [1],
            "s_p": [1],
            "M": 1,
            "U_pi": [[10, 10, 10]],
            "I_pi0": [[5, 0, 0]],
            "h_pi": [[0, 0, 0]],
            "C": 10,
            "f": 1,
            "a_ik": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
            "d_pit": [[[0], [3]]],
        }

        heuristic = MultProductProdctionRoutingProblemGreedyConstructiveHeuristic(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            rng=np.random.default_rng(123),
        )

        captured = {}

        class CaptureRoute:
            def greedyRandomizedConstruction(
                self, clients, demands, capacities, D, n_vehicles, alpha, rng
            ):
                captured["clients"] = list(clients)
                captured["demands"] = [list(demand) for demand in demands]
                return (
                    [[0, 1]],
                    0,
                    [{"veiculo": 0, "entregas": []}],
                    [[0, 1]],
                )

        heuristic.greedyRoute = CaptureRoute()
        heuristic.construirSolucao()

        self.assertEqual(captured["clients"], [0, 2])
        self.assertEqual(captured["demands"], [[0.0], [3]])

    def test_start_variables_casts_warm_start_values_to_float(self):
        mpprp_module = _load_mpprp_module()
        MultProductProdctionRoutingProblem = mpprp_module.MultProductProdctionRoutingProblem

        data = {
            "num_products": 1,
            "num_customers": 1,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 1,
            "b_p": [1],
            "c_p": [1],
            "s_p": [1],
            "M": 1,
            "U_pi": [[1, 1]],
            "I_pi0": [[0, 0]],
            "h_pi": [[0, 0]],
            "C": 1,
            "f": 1,
            "a_ik": [[0, 1], [1, 0]],
            "d_pit": [[[0]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

        solver = MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": True, "variables": {}},
        )
        solver.p = solver.i = solver.k = solver.t = solver.v = 1
        solver.model = FakeModel()
        solver.start = {
            "variables": {
                "X": [[np.int64(1)]],
                "Y": [[np.int64(1)]],
                "I": [[[np.int64(0)]]],
                "R": [[[[[np.int64(0)]]]]],
                "Q": [[[[np.int64(0)]]]],
                "Z": [[[[np.int64(0)]]]],
            }
        }
        solver.startVariables()

        self.assertIsNotNone(solver.model.added)
        self.assertTrue(solver.model.added.values)
        self.assertTrue(all(isinstance(value, float) for value in solver.model.added.values))


if __name__ == "__main__":
    unittest.main()
