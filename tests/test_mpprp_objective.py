import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_real_mpprp():
    module_name = "mpprp_real_objective_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    module_path = project_root / "src/solvers/MultProductProdctionRoutingProblem.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class DummyLogger:
    def info(self, message):
        pass

    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class MPPRPObjectiveTestCase(unittest.TestCase):
    @staticmethod
    def _data(num_customers=1):
        nodes = num_customers + 1
        return {
            "num_products": 1,
            "num_customers": num_customers,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 100,
            "U_pi": [[20] * nodes],
            "I_pi0": [[0] * nodes],
            "h_pi": [[0] + [1] * num_customers],
            "C": 10,
            "f": 7,
            "a_ik": [[0 if i == k else 11 for k in range(nodes)] for i in range(nodes)],
            "d_pit": [[[5] for _ in range(num_customers)]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

    def test_singleobjective_objective_builds_without_list_expression_error(self):
        data = self._data()

        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()

        with patch("mpprp_real_objective_test.Config.get_nested") as get_nested:
            values = {
                ("postprocessing", "build_target"): False,
                ("solver", "multiobjective"): False,
            }
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver.crateObjectiveFunction()

        self.assertIsNotNone(solver.model.objective_expr)

    def test_decision_variables_exclude_diagonal_arcs(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()

        self.assertEqual(6, len(solver.model.Z_v_i_k_t))
        self.assertEqual(6, len(solver.model.R_p_v_i_k_t))
        for node in range(3):
            self.assertNotIn((0, node, node, 0), solver.model.Z_v_i_k_t)
            self.assertNotIn((0, 0, node, node, 0), solver.model.R_p_v_i_k_t)

    def test_mip_start_ignores_diagonal_arc_values(self):
        module = _load_real_mpprp()
        nodes = 3
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2),
            dir="/tmp",
            log=DummyLogger(),
            start={
                "start": True,
                "variables": {
                    "X": [[0]],
                    "Y": [[0]],
                    "I": [[[0] for _ in range(nodes)]],
                    "Q": [[[[0] for _ in range(nodes)]]],
                    "R": [[[[[0] for _ in range(nodes)] for _ in range(nodes)]]],
                    "Z": [[[[0] for _ in range(nodes)] for _ in range(nodes)]],
                },
            },
        )
        solver.createDecisionVariables()

        solver.startVariables()

        self.assertEqual(1, solver.model.number_of_mip_starts)


if __name__ == "__main__":
    unittest.main()
