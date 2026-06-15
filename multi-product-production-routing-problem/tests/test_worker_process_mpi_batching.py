import sys
import time
import types
import unittest
from unittest.mock import patch

graph_display_module = types.ModuleType("src.helpers.GraphDisplay")
graph_display_module.graphResults = lambda *args, **kwargs: None
sys.modules["src.helpers.GraphDisplay"] = graph_display_module

targets_loader_module = types.ModuleType("src.helpers.TargetsLoader")
targets_loader_module.load_targets_by_file = lambda *args, **kwargs: {}
targets_loader_module.normalize_instance_file_key = lambda value: value
sys.modules["src.helpers.TargetsLoader"] = targets_loader_module

process_results_module = types.ModuleType("src.process.ProcessResults")
process_results_module.getResults = lambda *args, **kwargs: {"periods": []}
process_results_module.new_get_results = lambda *args, **kwargs: {}
sys.modules["src.process.ProcessResults"] = process_results_module

solver_module = types.ModuleType("src.solvers.MultProductProdctionRoutingProblem")
solver_module.MultProductProdctionRoutingProblem = object
sys.modules["src.solvers.MultProductProdctionRoutingProblem"] = solver_module

heuristic_module = types.ModuleType(
    "src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic"
)
heuristic_module.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic = object
sys.modules[
    "src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic"
] = heuristic_module

from src.process.WorkerProcess import WorkerProcess


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeFuture:
    def result(self):
        return None


class FakeExecutor:
    max_workers_used = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        FakeExecutor.max_workers_used.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return FakeFuture()


class WorkerProcessMPIBatchingTestCase(unittest.TestCase):
    def setUp(self):
        self.logger = DummyLogger()
        FakeExecutor.max_workers_used = []

    def _build_worker(self):
        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested:
            values = {
                ("workers", "mpi_batch_multiplier"): 1,
                ("workers", "mpi_batch_max_tasks"): 256,
                ("workers", "mpi_submit_guardrail_enabled"): True,
                ("workers", "mpi_heavy_instance_cap"): 32,
                ("postprocessing", "build_target"): False,
                ("postprocessing", "output"): None,
            }
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            return WorkerProcess(
                numWorkers="auto",
                timeSupervisor=1,
                log={"instancia": self.logger, "dirLogs": "./out/logs"},
                guardrail_runtime={"job_start_time": time.time(), "job_id": "local"},
            )

    def test_compute_mpi_batch_size_respects_multiplier_and_cap(self):
        worker = self._build_worker()
        with patch.object(worker, "_resolve_num_workers", return_value=256):
            batch_size, effective_workers = worker._compute_mpi_batch_size(1800)

        self.assertEqual(effective_workers, 256)
        self.assertEqual(batch_size, 256)

    def test_heavy_batch_uses_configured_cap(self):
        worker = self._build_worker()
        batch = [
            {"instancie": {"file": "./data/DATA_PRP_30C/PRP1.dat"}},
            {"instancie": {"file": "./data/DATA_PRP_30C/PRP2.dat"}},
        ]

        with patch.object(worker, "_resolve_num_workers", return_value=256):
            plan = worker._compute_batch_execution_plan(batch, total_tasks=1800)

        self.assertTrue(plan["heavy_batch"])
        self.assertEqual(plan["executor_workers"], 2)
        self.assertEqual(plan["heavy_instance_cap"], 32)

    def test_light_batch_does_not_use_heavy_cap(self):
        worker = self._build_worker()
        batch = [{"instancie": {"file": "./data/DATA_PRP_5C/PRP1.dat"}}]

        with patch.object(worker, "_resolve_num_workers", return_value=128):
            plan = worker._compute_batch_execution_plan(batch, total_tasks=12)

        self.assertFalse(plan["heavy_batch"])
        self.assertIsNone(plan["heavy_instance_cap"])

    def test_run_parallel_splits_large_task_set_into_multiple_batches(self):
        worker = self._build_worker()
        instancies = [
            {
                "file": f"./data/DATA_PRP_30C/PRP{i}.dat",
                "output": "./out/",
                "isPloat": False,
                "numThreads": 1,
                "timeLimit": 10,
            }
            for i in range(150)
        ]

        with patch("src.process.WorkerProcess.MPI_BOOL", True):
            with patch("src.process.WorkerProcess.MPIPoolExecutor", FakeExecutor, create=True):
                with patch("src.process.WorkerProcess.as_completed", side_effect=lambda futures: list(futures)):
                    with patch("src.process.WorkerProcess.process", return_value=None):
                        with patch.object(worker, "_resolve_num_workers", return_value=256):
                            worker.run_parallel(instancies=instancies, solver="GUROBY")

        batch_plan_logs = [
            message
            for level, message in self.logger.messages
            if level == "info" and "MPI_BATCH_PLAN" in message
        ]
        batch_start_logs = [
            message
            for level, message in self.logger.messages
            if level == "info" and "MPI_BATCH_START" in message
        ]

        self.assertGreater(len(batch_plan_logs), 1)
        self.assertGreater(len(batch_start_logs), 1)
        self.assertTrue(any("batch=1/8" in message for message in batch_plan_logs))
        self.assertTrue(any("batch=8/8" in message for message in batch_plan_logs))
        self.assertTrue(any("heavy_instance_cap=32" in message for message in batch_plan_logs))
        self.assertTrue(all(max_workers <= 32 for max_workers in FakeExecutor.max_workers_used))
        self.assertTrue(all(max_workers == 32 for max_workers in FakeExecutor.max_workers_used[:-1]))
        self.assertEqual(FakeExecutor.max_workers_used[-1], 8)


if __name__ == "__main__":
    unittest.main()
