import sys
import time
import types
import unittest
from unittest.mock import patch

_original_graph_display_module = sys.modules.get("src.helpers.GraphDisplay")
graph_display_module = types.ModuleType("src.helpers.GraphDisplay")
graph_display_module.graphResults = lambda *args, **kwargs: None
sys.modules.setdefault("src.helpers.GraphDisplay", graph_display_module)

_original_targets_loader_module = sys.modules.get("src.helpers.TargetsLoader")
targets_loader_module = types.ModuleType("src.helpers.TargetsLoader")
targets_loader_module.load_targets_by_file = lambda *args, **kwargs: {}
targets_loader_module.normalize_instance_file_key = lambda value: value
sys.modules.setdefault("src.helpers.TargetsLoader", targets_loader_module)

_original_process_results_module = sys.modules.get("src.process.ProcessResults")
process_results_module = types.ModuleType("src.process.ProcessResults")
process_results_module.getResults = lambda *args, **kwargs: {"periods": []}
process_results_module.new_get_results = lambda *args, **kwargs: {}
sys.modules.setdefault("src.process.ProcessResults", process_results_module)

_original_solver_module = sys.modules.get("src.solvers.MultProductProdctionRoutingProblem")
solver_module = types.ModuleType("src.solvers.MultProductProdctionRoutingProblem")
solver_module.MultProductProdctionRoutingProblem = object
sys.modules.setdefault("src.solvers.MultProductProdctionRoutingProblem", solver_module)

_original_pso_module = sys.modules.get(
    "src.solvers.ParticleSwarmOptimization"
)
pso_module = types.ModuleType(
    "src.solvers.ParticleSwarmOptimization"
)
pso_module.ParticleSwarmOptimization = object
sys.modules.setdefault(
    "src.solvers.ParticleSwarmOptimization",
    pso_module,
)

from src.process.InstanceProcess import InstanceProcess

if _original_graph_display_module is None:
    sys.modules.pop("src.helpers.GraphDisplay", None)
else:
    sys.modules["src.helpers.GraphDisplay"] = _original_graph_display_module

if _original_targets_loader_module is None:
    sys.modules.pop("src.helpers.TargetsLoader", None)
else:
    sys.modules["src.helpers.TargetsLoader"] = _original_targets_loader_module

if _original_process_results_module is None:
    sys.modules.pop("src.process.ProcessResults", None)
else:
    sys.modules["src.process.ProcessResults"] = _original_process_results_module

if _original_solver_module is None:
    sys.modules.pop("src.solvers.MultProductProdctionRoutingProblem", None)
else:
    sys.modules["src.solvers.MultProductProdctionRoutingProblem"] = _original_solver_module

if _original_pso_module is None:
    sys.modules.pop("src.solvers.ParticleSwarmOptimization", None)
else:
    sys.modules[
        "src.solvers.ParticleSwarmOptimization"
    ] = _original_pso_module


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeSolverInstance:
    def __init__(self):
        self.terminated = False

    def solver(self, timeLimit=None, numThreads=None):
        time.sleep(0.03)

    def getResults(self):
        return (
            [[[ [0.0] ]]],
            [[0.0]],
            [[0.0]],
            [[[0.0]]],
            [[[[[0.0]]]]],
            [[[[0.0]]]],
            [[0.0, 0.0, 0.0, 0.0, 0.0]],
            0.0,
            0.0,
            0.0,
            None,
            0,
            0.0,
            0,
            0.0,
            [],
        )

    def terminate(self):
        self.terminated = True


class InstanceProcessWatchdogTestCase(unittest.TestCase):
    def _build_process(self, logger):
        return InstanceProcess(
            instance="./data/test.dat",
            output="./out/test/",
            log=logger,
            solver="GUROBY",
            guardrail_runtime={
                "job_start_time": time.time(),
                "mpi_rank": "unknown",
                "pid": 100,
                "ppid": 10,
            },
            task_context={"mpi_batch": 2, "task_number": 9},
        )

    def test_watchdog_logs_during_solve(self):
        logger = DummyLogger()
        process = self._build_process(logger)
        fake_instance = FakeSolverInstance()
        fake_data = {
            "file": "./data/test.dat",
            "weight": [0.2] * 5,
            "alpha": 0.01,
            "s_p": [0.0],
            "c_p": [0.0],
            "h_pi": [[0.0]],
            "f": [[0.0]],
            "a_ik": [[0.0]],
            "num_periods": 1,
            "coordXY": {"x": [0.0], "y": [0.0]},
            "I_pi0": [[0.0]],
            "d_pit": [[[0.0]]],
        }

        with patch("src.process.InstanceProcess.RD") as mock_rd:
            mock_rd.return_value.getDataSet.return_value = fake_data
            with patch.object(process, "solverInstancie", return_value=fake_instance):
                with patch("src.process.InstanceProcess.getResults", return_value={"periods": []}):
                    with patch.object(process.guardrails, "solve_watchdog_interval_seconds", 0.01):
                        process.process()

        watchdog_logs = [
            message for _, message in logger.messages if "event=solve_watchdog" in message
        ]
        identity_logs = [
            message for _, message in logger.messages if "event=instance_identity" in message
        ]
        checkpoint_logs = [
            message for _, message in logger.messages if "event=memory_checkpoint" in message
        ]

        self.assertTrue(watchdog_logs)
        self.assertTrue(identity_logs)
        self.assertTrue(checkpoint_logs)

    def test_watchdog_stops_on_solver_exception(self):
        logger = DummyLogger()
        process = self._build_process(logger)

        class BrokenSolverInstance(FakeSolverInstance):
            def solver(self, timeLimit=None, numThreads=None):
                time.sleep(0.02)
                raise RuntimeError("solver boom")

        fake_data = {
            "file": "./data/test.dat",
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

        with patch("src.process.InstanceProcess.RD") as mock_rd:
            mock_rd.return_value.getDataSet.return_value = fake_data
            with patch.object(process, "solverInstancie", return_value=BrokenSolverInstance()):
                with patch.object(process.guardrails, "solve_watchdog_interval_seconds", 0.01):
                    with self.assertRaises(RuntimeError):
                        process.process()

        self.assertIsNone(process._solve_watchdog_thread)
        self.assertIsNone(process._solve_watchdog_stop)


if __name__ == "__main__":
    unittest.main()
