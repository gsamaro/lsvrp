import os
import resource
import socket
import time
from contextlib import suppress
from dataclasses import dataclass

from config import Config


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_proc_status():
    status = {}
    proc_path = "/proc/self/status"
    if not os.path.exists(proc_path):
        return status

    try:
        with open(proc_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                status[key.strip()] = value.strip()
    except OSError:
        return {}

    return status


def _parse_proc_kb(raw_value):
    if not raw_value:
        return None

    parts = raw_value.split()
    if not parts:
        return None

    try:
        return int(parts[0])
    except ValueError:
        return None


@dataclass
class GuardrailSnapshot:
    status: str
    reasons: list
    elapsed_seconds: float
    walltime_seconds: int
    walltime_ratio: float
    rss_gb: float
    peak_rss_gb: float
    memory_limit_gb: float
    memory_ratio: float
    approximate_memory: bool
    should_stop: bool
    should_warn: bool
    job_id: str
    host: str
    mpi_size: int
    mpi_rank: str
    pid: int
    ppid: int


class JobGuardrails:
    DEFAULT_WALLTIME_SECONDS = 3 * 24 * 60 * 60
    DEFAULT_MEMORY_PER_NODE_GB = 488.0
    DEFAULT_WARNING_RATIO = 0.85
    DEFAULT_STOP_RATIO = 0.95
    DEFAULT_CHECK_INTERVAL_SECONDS = 60
    DEFAULT_SOLVE_WATCHDOG_INTERVAL_SECONDS = 60

    def __init__(self, config=None, runtime_context=None):
        cfg = config or {}
        runtime = runtime_context or {}

        self.enabled = bool(cfg.get("enabled", False))
        self.walltime_seconds = _safe_int(
            cfg.get("walltime_seconds"), self.DEFAULT_WALLTIME_SECONDS
        )
        self.walltime_warning_ratio = _safe_float(
            cfg.get("walltime_warning_ratio"), self.DEFAULT_WARNING_RATIO
        )
        self.walltime_stop_ratio = _safe_float(
            cfg.get("walltime_stop_ratio"), self.DEFAULT_STOP_RATIO
        )
        self.memory_per_node_gb = _safe_float(
            cfg.get("memory_per_node_gb"), self.DEFAULT_MEMORY_PER_NODE_GB
        )
        self.memory_warning_ratio = _safe_float(
            cfg.get("memory_warning_ratio"), self.DEFAULT_WARNING_RATIO
        )
        self.memory_stop_ratio = _safe_float(
            cfg.get("memory_stop_ratio"), self.DEFAULT_STOP_RATIO
        )
        self.check_interval_seconds = _safe_int(
            cfg.get("check_interval_seconds"), self.DEFAULT_CHECK_INTERVAL_SECONDS
        )
        self.log_solver_phase_timing = bool(cfg.get("log_solver_phase_timing", True))
        self.solve_watchdog_enabled = bool(cfg.get("solve_watchdog_enabled", True))
        self.solve_watchdog_interval_seconds = _safe_int(
            cfg.get("solve_watchdog_interval_seconds"),
            self.DEFAULT_SOLVE_WATCHDOG_INTERVAL_SECONDS,
        )
        self.log_memory_checkpoints = bool(cfg.get("log_memory_checkpoints", True))
        self.log_rank_identity = bool(cfg.get("log_rank_identity", True))

        self.job_start_time = _safe_float(runtime.get("job_start_time"), time.time())
        self.job_id = runtime.get("job_id") or os.environ.get("PBS_JOBID") or "local"
        self.host = runtime.get("host") or socket.gethostname()
        self.mpi_size = _safe_int(runtime.get("mpi_size"), self._detect_mpi_size())
        self.mpi_rank = str(runtime.get("mpi_rank") or self._detect_mpi_rank())
        self.pid = _safe_int(runtime.get("pid"), os.getpid())
        self.ppid = _safe_int(runtime.get("ppid"), os.getppid())
        self.last_logged_at = 0.0

    @classmethod
    def from_config(cls, runtime_context=None):
        return cls(Config.get("guardrails", default={}) or {}, runtime_context)

    @classmethod
    def build_runtime_context(cls):
        return {
            "job_start_time": time.time(),
            "job_id": os.environ.get("PBS_JOBID") or "local",
            "host": socket.gethostname(),
            "mpi_size": cls._discover_mpi_size(),
            "mpi_rank": cls._discover_mpi_rank(),
            "pid": os.getpid(),
            "ppid": os.getppid(),
        }

    @staticmethod
    def _discover_mpi_size():
        for env_name in ("PBS_NP", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
            value = os.environ.get(env_name)
            if value:
                return _safe_int(value, 1)

        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            try:
                with open(nodefile, "r", encoding="utf-8") as handle:
                    return sum(1 for line in handle if line.strip())
            except OSError:
                return 1

        return 1

    def _detect_mpi_size(self):
        return self._discover_mpi_size()

    @staticmethod
    def _discover_mpi_rank():
        for env_name in (
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "PMIX_RANK",
            "SLURM_PROCID",
            "MPI_RANK",
        ):
            value = os.environ.get(env_name)
            if value is not None:
                return str(value)

        with suppress(Exception):
            from mpi4py import MPI

            if MPI.Is_initialized():
                return str(MPI.COMM_WORLD.Get_rank())

        return "unknown"

    def _detect_mpi_rank(self):
        return self._discover_mpi_rank()

    def is_enabled(self):
        return self.enabled

    def elapsed_seconds(self):
        return max(0.0, time.time() - self.job_start_time)

    def _memory_usage_gb(self):
        proc_status = _read_proc_status()
        rss_kb = _parse_proc_kb(proc_status.get("VmRSS"))
        peak_kb = _parse_proc_kb(proc_status.get("VmHWM"))

        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss = getattr(usage, "ru_maxrss", 0) or 0

        if peak_kb is None:
            peak_kb = int(ru_maxrss)

        if rss_kb is None:
            rss_kb = int(ru_maxrss)

        approximate = bool(not proc_status)
        rss_gb = rss_kb / (1024 * 1024)
        peak_rss_gb = peak_kb / (1024 * 1024)
        return rss_gb, peak_rss_gb, approximate

    def snapshot(self):
        elapsed = self.elapsed_seconds()
        walltime_ratio = (
            elapsed / self.walltime_seconds if self.walltime_seconds > 0 else 0.0
        )
        rss_gb, peak_rss_gb, approximate = self._memory_usage_gb()
        memory_ratio = (
            rss_gb / self.memory_per_node_gb if self.memory_per_node_gb > 0 else 0.0
        )

        status = "ok"
        reasons = []

        if walltime_ratio >= self.walltime_stop_ratio:
            status = "stop"
            reasons.append("walltime")
        elif walltime_ratio >= self.walltime_warning_ratio:
            if status != "stop":
                status = "warning"
            reasons.append("walltime")

        if memory_ratio >= self.memory_stop_ratio:
            status = "stop"
            reasons.append("memory")
        elif memory_ratio >= self.memory_warning_ratio:
            if status != "stop":
                status = "warning"
            reasons.append("memory")

        return GuardrailSnapshot(
            status=status,
            reasons=reasons,
            elapsed_seconds=elapsed,
            walltime_seconds=self.walltime_seconds,
            walltime_ratio=walltime_ratio,
            rss_gb=rss_gb,
            peak_rss_gb=peak_rss_gb,
            memory_limit_gb=self.memory_per_node_gb,
            memory_ratio=memory_ratio,
            approximate_memory=approximate,
            should_stop=status == "stop",
            should_warn=status in ("warning", "stop"),
            job_id=self.job_id,
            host=self.host,
            mpi_size=self.mpi_size,
            mpi_rank=self.mpi_rank,
            pid=self.pid,
            ppid=self.ppid,
        )

    def should_emit_log(self, force=False):
        if force:
            self.last_logged_at = time.time()
            return True

        now = time.time()
        if now - self.last_logged_at >= max(1, self.check_interval_seconds):
            self.last_logged_at = now
            return True
        return False

    def format_snapshot(self, snapshot, context=None):
        context_bits = []
        context = context or {}
        for key in (
            "event",
            "mpi_batch",
            "task_number",
            "instance_file",
            "completed",
            "total",
            "phase",
            "checkpoint",
            "watchdog_seq",
        ):
            value = context.get(key)
            if value is not None:
                context_bits.append(f"{key}={value}")

        memory_note = "approx-process-rss" if snapshot.approximate_memory else "process-rss"
        context_bits.extend(
            [
                f"job_id={snapshot.job_id}",
                f"host={snapshot.host}",
                f"mpi_size={snapshot.mpi_size}",
                f"mpi_rank={snapshot.mpi_rank}",
                f"pid={snapshot.pid}",
                f"ppid={snapshot.ppid}",
                f"elapsed={snapshot.elapsed_seconds:.1f}s",
                f"walltime={snapshot.walltime_seconds}s",
                f"walltime_ratio={snapshot.walltime_ratio:.3f}",
                f"rss_gb={snapshot.rss_gb:.3f}",
                f"peak_rss_gb={snapshot.peak_rss_gb:.3f}",
                f"memory_limit_gb={snapshot.memory_limit_gb:.1f}",
                f"memory_ratio={snapshot.memory_ratio:.3f}",
                f"memory_scope={memory_note}",
                f"status={snapshot.status}",
            ]
        )
        if snapshot.reasons:
            context_bits.append(f"reasons={','.join(snapshot.reasons)}")
        return "guardrail " + " ".join(context_bits)
