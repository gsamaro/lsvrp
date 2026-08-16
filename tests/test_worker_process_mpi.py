import unittest
from unittest.mock import MagicMock, patch

from src.process.WorkerProcess import WorkerProcess, process
from constants import ALPHA, WEIGHTS_OPTIMIZE, WEIGHTS_TARGET


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def debug(self, message):
        self.messages.append(("debug", message))

    def warning(self, message):
        self.messages.append(("warning", message))

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
        return WorkerProcess(numWorkers="auto", log=self.logger)

    @staticmethod
    def _config_value(build_target):
        return lambda *keys, default=None: {
            ("postprocessing", "output"): None,
            ("postprocessing", "build_target"): build_target,
        }.get(keys, default)

    @staticmethod
    def _instance(number):
        return {
            "file": f"./data/DATA_PRP_5C/PRP{number}.dat",
            "output": "./out/",
            "numThreads": 1,
            "timeLimit": 10,
        }

    def test_mpi_uses_one_executor_and_submits_all_tasks(self):
        worker = self._build_worker()
        instancies = [self._instance(1), self._instance(2)]
        total_tasks = len(instancies) * len(WEIGHTS_OPTIMIZE) * len(ALPHA)

        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested, patch("src.process.WorkerProcess.MPI_BOOL", True):
            get_nested.side_effect = self._config_value(False)
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

        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested, patch("src.process.WorkerProcess.MPI_BOOL", False):
            get_nested.side_effect = self._config_value(False)
            with patch("src.process.WorkerProcess.process", side_effect=fake_process):
                worker.run_parallel(instancies=[self._instance(1)], solver="PSO")

        self.assertEqual(len(captured), len(WEIGHTS_OPTIMIZE) * len(ALPHA))
        self.assertTrue(all(solver == "PSO" for solver, _ in captured))
        self.assertTrue(
            all(
                set(context) == {"file", "weight", "alpha", "run_tag", "task_number", "label"}
                for _, context in captured
            )
        )

    def test_run_parallel_resolves_target_weights_without_reloading_module(self):
        worker = self._build_worker()
        captured = []

        def fake_process(log, instancie, solver, w, targets_by_file, alpha, context=None):
            captured.append(w)

        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested, patch(
            "src.process.WorkerProcess.MPI_BOOL", False
        ), patch("src.process.WorkerProcess.process", side_effect=fake_process):
            get_nested.side_effect = self._config_value(True)
            worker.run_parallel(instancies=[self._instance(1)])

        self.assertEqual(len(captured), len(WEIGHTS_TARGET) * len(ALPHA))
        self.assertEqual(captured[0], WEIGHTS_TARGET[0])

    def test_process_instantiates_instance_process(self):
        instancie = self._instance(1)
        context = {
            "label": "arquivo: ./data/DATA_PRP_5C/PRP1.dat | peso: [0.2] | alpha: 0.01"
        }
        instance_process = MagicMock()

        with patch("src.process.WorkerProcess.InstanceProcess", instance_process):
            process(
                self.logger,
                instancie,
                "PSO",
                [0.2],
                {"PRP1.dat": {}},
                0.01,
                context,
            )

        instance_process.assert_called_once_with(
            instancie["file"],
            instancie["output"],
            timeLimit=instancie["timeLimit"],
            numThreads=instancie["numThreads"],
            log=self.logger,
            solver="PSO",
            weight=[0.2],
            targets_by_file={"PRP1.dat": {}},
            alpha=0.01,
            task_context=context,
        )
        instance_process.return_value.process.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
