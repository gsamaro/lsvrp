from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.process.RuntimeContext import (
    RuntimeContext,
    build_instances,
    build_runtime_context,
    finalize_postprocessing,
    is_supported_prp_file,
    validate_runtime_context,
)


class DummyConfig:
    values = {
        ("solver", "threadsLimit"): "None",
        ("solver", "timeLimit"): "10",
        ("workers", "num"): "auto",
        ("instance", "output"): "./out/",
        ("instance", "dir"): "./data/",
        ("instance", "files"): ["DATA_PRP_5C"],
        ("solver", "method"): "PSO",
        ("postprocessing", "build_target"): False,
        ("solver", "multiobjective"): False,
    }

    @classmethod
    def get_nested(cls, *keys):
        return cls.values[keys]


class DummyLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class RuntimeContextTestCase(unittest.TestCase):
    def test_build_runtime_context_normalizes_threads_limit(self):
        with patch("src.process.RuntimeContext._build_run_tag", return_value="test-run"):
            context = build_runtime_context(
                config=DummyConfig,
                now=datetime(2026, 8, 16, tzinfo=timezone.utc),
            )

        self.assertEqual(context.run_tag, "test-run")
        self.assertIsNone(context.threads_limit)
        self.assertEqual(context.time_limit, 10)
        self.assertEqual(context.method, "PSO")

    def test_build_instances_from_directory_skips_prp31_and_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            instance_dir = root / "data" / "DATA_PRP_5C"
            output_dir = root / "out"
            instance_dir.mkdir(parents=True)
            for file_name in ["PRP30_C5.dat", "PRP31_C5.dat", "other.dat"]:
                (instance_dir / file_name).touch()

            instances = build_instances(
                f"{root}/data/", f"{output_dir}/", ["DATA_PRP_5C"], 1, 10
            )

            self.assertEqual(
                {item["file"].split("/")[-1] for item in instances},
                {"PRP30_C5.dat", "other.dat"},
            )
            self.assertFalse((output_dir / "DATA_PRP_5C" / "PRP31_C5").exists())

    def test_build_instances_accepts_single_dat_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            instances = build_instances(
                f"{root}/data/", f"{root}/out/", ["DATA_PRP_5C/PRP1.dat"], 2, 30
            )

            self.assertEqual(instances[0]["file"], f"{root}/data/DATA_PRP_5C/PRP1.dat")
            self.assertEqual(instances[0]["numThreads"], 2)
            self.assertEqual(instances[0]["timeLimit"], 30)

    def test_prp_filter_preserves_malformed_and_non_prp_names(self):
        self.assertTrue(is_supported_prp_file("other.dat"))
        self.assertTrue(is_supported_prp_file("PRP31.dat"))
        self.assertFalse(is_supported_prp_file("PRP31_C5.dat"))

    def test_validate_runtime_context_rejects_target_with_multiobjective(self):
        context = RuntimeContext(
            "tag", None, 10, 1, "./out/", "./data/", [], "PSO", True, True
        )
        log = DummyLogger()

        with self.assertRaisesRegex(RuntimeError, "Build target"):
            validate_runtime_context(context, log)

        self.assertTrue(log.errors)

    def test_finalize_postprocessing_raises_without_consolidation(self):
        postprocessing = MagicMock()
        postprocessing.log = DummyLogger()
        postprocessing.union_results.return_value = None

        with self.assertRaisesRegex(RuntimeError, "consolidação"):
            finalize_postprocessing(postprocessing, "run", True)

        postprocessing.build_target.assert_not_called()

    def test_finalize_postprocessing_builds_target_after_consolidation(self):
        postprocessing = MagicMock()
        postprocessing.union_results.return_value = "./out/union_results.xlsx"

        union_path = finalize_postprocessing(postprocessing, "run", True)

        self.assertEqual(union_path, "./out/union_results.xlsx")
        postprocessing.build_target.assert_called_once_with(
            union_results_path="./out/union_results.xlsx"
        )


if __name__ == "__main__":
    unittest.main()
