import unittest
from unittest.mock import MagicMock, patch

import main
from src.process.RuntimeContext import RuntimeContext


class MainImportTestCase(unittest.TestCase):
    def test_main_imports_without_missing_helper_modules(self):
        self.assertIsNotNone(main)

    def test_main_delegates_runtime_flow(self):
        context = RuntimeContext(
            run_tag="test-run",
            threads_limit=1,
            time_limit=10,
            workers="auto",
            output="./out/",
            instance_dir="./data/",
            files=["DATA_PRP_5C"],
            method="PSO",
            build_target=False,
            multiobjective=False,
        )
        log = MagicMock()
        instances = [{"file": "./data/DATA_PRP_5C/PRP1.dat"}]
        postprocessing = MagicMock()

        with patch("main.build_runtime_context", return_value=context), patch(
            "main.create_run_logger", return_value=log
        ), patch("main.validate_runtime_context") as validate, patch(
            "main.build_instances", return_value=instances
        ) as build_instances, patch("main.WorkerProcess") as worker_process, patch(
            "main.PostProcessingProcess", return_value=postprocessing
        ), patch("main.finalize_postprocessing") as finalize:
            main.main()

        validate.assert_called_once_with(context, log)
        build_instances.assert_called_once_with(
            "./data/", "./out/", ["DATA_PRP_5C"], 1, 10
        )
        worker_process.assert_called_once_with("auto", log, run_tag="test-run")
        worker_process.return_value.run_parallel.assert_called_once_with(
            instancies=instances, solver="PSO"
        )
        finalize.assert_called_once_with(postprocessing, "test-run", False)


if __name__ == "__main__":
    unittest.main()
