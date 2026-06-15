import tempfile
import time
import unittest
from unittest.mock import patch

from src.helpers.JobGuardrails import JobGuardrails


class JobGuardrailsTestCase(unittest.TestCase):
    def test_walltime_stop_threshold(self):
        guardrails = JobGuardrails(
            config={
                "enabled": True,
                "walltime_seconds": 100,
                "walltime_warning_ratio": 0.5,
                "walltime_stop_ratio": 0.8,
                "memory_per_node_gb": 1000,
            },
            runtime_context={"job_start_time": time.time() - 90},
        )

        snapshot = guardrails.snapshot()

        self.assertEqual(snapshot.status, "stop")
        self.assertIn("walltime", snapshot.reasons)

    def test_memory_warning_threshold(self):
        guardrails = JobGuardrails(
            config={
                "enabled": True,
                "walltime_seconds": 1000,
                "memory_per_node_gb": 2,
                "memory_warning_ratio": 0.2,
                "memory_stop_ratio": 0.9,
            },
            runtime_context={"job_start_time": time.time()},
        )

        with patch.object(guardrails, "_memory_usage_gb", return_value=(0.6, 0.8, False)):
            snapshot = guardrails.snapshot()

        self.assertEqual(snapshot.status, "warning")
        self.assertIn("memory", snapshot.reasons)

    def test_build_runtime_context_reads_pbs_nodefile(self):
        with tempfile.NamedTemporaryFile("w", delete=True) as handle:
            handle.write("nodeA\nnodeA\nnodeB\n")
            handle.flush()
            with patch.dict("os.environ", {"PBS_NODEFILE": handle.name}, clear=False):
                self.assertEqual(JobGuardrails._discover_mpi_size(), 3)

    def test_discover_mpi_rank_from_env(self):
        with patch.dict("os.environ", {"OMPI_COMM_WORLD_RANK": "7"}, clear=False):
            self.assertEqual(JobGuardrails._discover_mpi_rank(), "7")

    def test_missing_proc_status_falls_back_to_resource(self):
        guardrails = JobGuardrails(
            config={"enabled": True},
            runtime_context={"job_start_time": time.time()},
        )

        with patch("src.helpers.JobGuardrails.os.path.exists", return_value=False):
            with patch.object(guardrails, "_memory_usage_gb") as memory_mock:
                memory_mock.return_value = (0.1, 0.2, True)
                snapshot = guardrails.snapshot()

        self.assertTrue(snapshot.approximate_memory)
        self.assertGreaterEqual(snapshot.peak_rss_gb, snapshot.rss_gb)

    def test_format_snapshot_includes_rank_and_pid(self):
        guardrails = JobGuardrails(
            config={"enabled": True},
            runtime_context={
                "job_start_time": time.time(),
                "mpi_rank": "3",
                "pid": 123,
                "ppid": 45,
            },
        )

        with patch.object(guardrails, "_memory_usage_gb", return_value=(0.1, 0.2, False)):
            snapshot = guardrails.snapshot()

        message = guardrails.format_snapshot(
            snapshot,
            context={
                "event": "solve_watchdog",
                "phase": "solve",
                "checkpoint": "before_solve",
                "watchdog_seq": 2,
            },
        )

        self.assertIn("mpi_rank=3", message)
        self.assertIn("pid=123", message)
        self.assertIn("ppid=45", message)
        self.assertIn("watchdog_seq=2", message)

    def test_refresh_runtime_context_updates_local_identity(self):
        guardrails = JobGuardrails(
            config={"enabled": True},
            runtime_context={
                "job_start_time": time.time(),
                "mpi_rank": "0",
                "pid": 1,
                "ppid": 2,
                "host": "master",
            },
        )

        with patch("src.helpers.JobGuardrails.socket.gethostname", return_value="worker-a"):
            with patch("src.helpers.JobGuardrails.os.getpid", return_value=333):
                with patch("src.helpers.JobGuardrails.os.getppid", return_value=222):
                    with patch.dict("os.environ", {"OMPI_COMM_WORLD_RANK": "17"}, clear=False):
                        runtime = guardrails.refresh_runtime_context()

        self.assertEqual(runtime["host"], "worker-a")
        self.assertEqual(runtime["pid"], 333)
        self.assertEqual(runtime["ppid"], 222)
        self.assertEqual(runtime["mpi_rank"], "17")
        self.assertEqual(guardrails.host, "worker-a")
        self.assertEqual(guardrails.pid, 333)
        self.assertEqual(guardrails.ppid, 222)
        self.assertEqual(guardrails.mpi_rank, "17")


if __name__ == "__main__":
    unittest.main()
