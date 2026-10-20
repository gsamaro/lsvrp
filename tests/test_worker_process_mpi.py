import pickle
import unittest
from unittest.mock import MagicMock, patch

from docplex.mp.utils import DOcplexLimitsExceeded

from src.process.WorkerProcess import WorkerProcess, process, process_safely
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
        return WorkerProcess(
            numWorkers="auto",
            log=self.logger,
            commit_hash="012345",
            config_hash="a" * 64,
            config_json='{"solver":{"method":"PSO"}}',
        )

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
                set(args[-1])
                == {
                    "file",
                    "weight",
                    "alpha",
                    "run_tag",
                    "commit_hash",
                    "config_hash",
                    "config_json",
                    "task_number",
                    "label",
                }
                for _, args, _ in FakeExecutor.submitted
            )
        )
        self.assertTrue(
            all(
                args[-1]["commit_hash"]
                == "012345"
                for _, args, _ in FakeExecutor.submitted
            )
        )
        self.assertTrue(
            all(
                args[-1]["config_hash"] == "a" * 64
                and args[-1]["config_json"] == '{"solver":{"method":"PSO"}}'
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
                set(context)
                == {
                    "file",
                    "weight",
                    "alpha",
                    "run_tag",
                    "commit_hash",
                    "config_hash",
                    "config_json",
                    "task_number",
                    "label",
                }
                for _, context in captured
            )
        )
        self.assertTrue(
            all(
                context["commit_hash"]
                == "012345"
                for _, context in captured
            )
        )
        self.assertTrue(
            all(
                context["config_hash"] == "a" * 64
                and context["config_json"] == '{"solver":{"method":"PSO"}}'
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

    def test_process_applies_the_experiment_config_carried_by_the_task(self):
        """MPI workers only know config.json; the task carries its experiment config."""
        import json

        from config import Config
        from src.process import WorkerProcess as worker_module

        previous = Config._data
        worker_module._applied_config_hash = None
        try:
            Config.replace({"solver": {"timeLimit": 10, "polishing": {"after_seconds": None}}})
            experiment = Config.snapshot()
            experiment["solver"]["polishing"]["after_seconds"] = 3000
            context = {
                "label": "tarefa",
                "config_hash": "h1",
                "config_json": json.dumps(experiment),
            }
            seen = []

            def fake_instance_process(*args, **kwargs):
                seen.append(Config.get_nested("solver", "polishing", "after_seconds"))
                return MagicMock()

            with patch(
                "src.process.WorkerProcess.InstanceProcess", fake_instance_process
            ):
                process(self.logger, self._instance(1), "PSO", [0.2], {}, 0.99, context)
                Config.replace({"solver": {"timeLimit": 10}})
                # same hash: the already installed config is not applied again
                process(self.logger, self._instance(1), "PSO", [0.2], {}, 0.99, context)
                # a different experiment (new hash) is applied
                other = dict(context, config_hash="h2")
                process(self.logger, self._instance(1), "PSO", [0.2], {}, 0.99, other)

            self.assertEqual(seen, [3000, None, 3000])
        finally:
            Config._data = previous
            worker_module._applied_config_hash = None

    def test_process_without_config_json_keeps_the_local_config(self):
        from config import Config

        previous = Config._data
        try:
            Config.replace({"solver": {"timeLimit": 55}})
            with patch("src.process.WorkerProcess.InstanceProcess"):
                process(self.logger, self._instance(1), "PSO", [0.2], {}, 0.99, {"label": "x"})
            self.assertEqual(Config.get_nested("solver", "timeLimit"), 55)
        finally:
            Config._data = previous

    def test_shared_executor_is_reused_and_not_shut_down(self):
        worker = self._build_worker()
        executor = FakeExecutor(max_workers=2)
        executor.exited = False
        executor.__exit__ = lambda *a: setattr(executor, "exited", True)

        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested, patch(
            "src.process.WorkerProcess.MPI_BOOL", True
        ), patch(
            "src.process.WorkerProcess.as_completed", side_effect=lambda f: list(f)
        ), patch.object(worker, "_resolve_num_workers", return_value=2):
            get_nested.side_effect = self._config_value(False)
            with patch("src.process.WorkerProcess.MPIPoolExecutor", FakeExecutor, create=True):
                worker.run_parallel([self._instance(1)], "PSO", executor=executor)
                worker.run_parallel([self._instance(2)], "PSO", executor=executor)

        total = len(WEIGHTS_OPTIMIZE) * len(ALPHA)
        self.assertEqual(len(FakeExecutor.submitted), 2 * total)
        self.assertEqual(FakeExecutor.max_workers_used, [2])  # only the one we built
        self.assertFalse(executor.exited)

    def test_count_tasks_matches_what_run_parallel_creates(self):
        worker = self._build_worker()
        with patch("src.process.WorkerProcess.Config.get_nested") as get_nested:
            get_nested.side_effect = self._config_value(False)
            self.assertEqual(
                worker.count_tasks([self._instance(1), self._instance(2)]),
                2 * len(WEIGHTS_OPTIMIZE) * len(ALPHA),
            )

    def test_process_safely_serializes_docplex_limits_error(self):
        error = DOcplexLimitsExceeded(1, 2)

        with patch("src.process.WorkerProcess.process", side_effect=error):
            result = process_safely(
                self.logger,
                self._instance(1),
                "GUROBY",
                [0.2],
                {},
                0.99,
                {"label": "tarefa de teste"},
            )

        self.assertEqual(result["error_type"], "DOcplexLimitsExceeded")
        self.assertIn("Problem size limits", result["error_message"])
        self.assertIsInstance(result["traceback"], str)
        pickle.dumps(result)


if __name__ == "__main__":
    unittest.main()
