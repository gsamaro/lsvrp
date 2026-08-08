import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_real_mpprp():
    module_name = "mpprp_real_objective_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/solvers/MultProductProdctionRoutingProblem.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class DummyLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class MPPRPObjectiveTestCase(unittest.TestCase):
    def test_singleobjective_objective_builds_without_list_expression_error(self):
        data = {
            "num_products": 1,
            "num_customers": 1,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 100,
            "U_pi": [[20, 20]],
            "I_pi0": [[0, 0]],
            "h_pi": [[0, 1]],
            "C": 10,
            "f": 7,
            "a_ik": [[0, 11], [11, 0]],
            "d_pit": [[[5]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

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


if __name__ == "__main__":
    unittest.main()
