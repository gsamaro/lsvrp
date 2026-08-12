import sys
import types
import unittest
from unittest.mock import patch

_original_graph_display_module = sys.modules.get("src.helpers.GraphDisplay")
graph_display_module = types.ModuleType("src.helpers.GraphDisplay")
graph_display_module.graphResults = lambda *args, **kwargs: None
sys.modules["src.helpers.GraphDisplay"] = graph_display_module

_original_process_results_module = sys.modules.get("src.process.ProcessResults")
process_results_module = types.ModuleType("src.process.ProcessResults")
process_results_module.getResults = lambda *args, **kwargs: {"periods": []}
sys.modules["src.process.ProcessResults"] = process_results_module

_original_solver_module = sys.modules.get("src.solvers.MultProductProdctionRoutingProblem")
solver_module = types.ModuleType("src.solvers.MultProductProdctionRoutingProblem")
solver_module.MultProductProdctionRoutingProblem = object
sys.modules["src.solvers.MultProductProdctionRoutingProblem"] = solver_module

_original_pso_module = sys.modules.get("src.solvers.ParticleSwarmOptimization")
pso_module = types.ModuleType("src.solvers.ParticleSwarmOptimization")
pso_module.ParticleSwarmOptimization = object
sys.modules["src.solvers.ParticleSwarmOptimization"] = pso_module

from src.process.WorkerProcess import WorkerProcess, WEIGHTS
from constants import ALPHA

if _original_graph_display_module is None:
    sys.modules.pop("src.helpers.GraphDisplay", None)
else:
    sys.modules["src.helpers.GraphDisplay"] = _original_graph_display_module
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
    sys.modules["src.solvers.ParticleSwarmOptimization"] = _original_pso_module


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeFuture:
    def result(self):
        return None


class FakeExecutor:
    max_workers_used = []
    submitted = []

    def __init__(self, max_workers):
        FakeExecutor.max_workers_used.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        FakeExecutor.submitted.append((fn, args, kwargs))
        return FakeFuture()


class WorkerProcessMPITestCase(unittest.TestCase):
    def setUp(self):
        self.logger = DummyLogger()
        FakeExecutor.max_workers_used = []
        FakeExecutor.submitted = []

    def _build_worker(self):
        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: {
                ("postprocessing", "output"): None,
            }.get(keys, default)
            return WorkerProcess(
                numWorkers="auto",
                timeSupervisor=1,
                log={"instancia": self.logger, "dirLogs": "./out/logs"},
            )

    @staticmethod
    def _instance(number):
        return {
            "file": f"./data/DATA_PRP_5C/PRP{number}.dat",
            "output": "./out/",
            "isPloat": False,
            "numThreads": 1,
            "timeLimit": 10,
        }

    def test_mpi_uses_one_executor_and_submits_all_tasks(self):
        worker = self._build_worker()
        instancies = [self._instance(1), self._instance(2)]
        total_tasks = len(instancies) * len(WEIGHTS) * len(ALPHA)

        with patch("src.process.WorkerProcess.MPI_BOOL", True):
            with patch("src.process.WorkerProcess.MPIPoolExecutor", FakeExecutor, create=True):
                with patch("src.process.WorkerProcess.as_completed", side_effect=lambda futures: list(futures)):
                    with patch.object(worker, "_resolve_num_workers", return_value=3):
                        worker.run_parallel(instancies=instancies, solver="GUROBY")

        self.assertEqual(FakeExecutor.max_workers_used, [3])
        self.assertEqual(len(FakeExecutor.submitted), total_tasks)
        self.assertTrue(
            all(
                set(args[-1]) == {"file", "weight", "alpha", "run_tag", "task_number", "label"}
                for _, args, _ in FakeExecutor.submitted
            )
        )

    def test_sequential_fallback_preserves_solver(self):
        worker = self._build_worker()
        captured = []

        def fake_process(log, instancie, solver, w, targets_by_file, alpha, context=None):
            captured.append((solver, context))

        with patch("src.process.WorkerProcess.MPI_BOOL", False):
            with patch("src.process.WorkerProcess.process", side_effect=fake_process):
                worker.run_parallel(instancies=[self._instance(1)], solver="PSO")

        self.assertEqual(len(captured), len(WEIGHTS) * len(ALPHA))
        self.assertTrue(all(solver == "PSO" for solver, _ in captured))
        self.assertTrue(
            all(
                set(context) == {"file", "weight", "alpha", "run_tag", "task_number", "label"}
                for _, context in captured
            )
        )


if __name__ == "__main__":
    unittest.main()
