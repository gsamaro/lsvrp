import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.helpers.SolverTelemetry import write_shards


def _summary(run_id, strategy, first=None, gap=None, timed_out=False):
    return {
        "run_id": run_id, "experiment_id": "test", "instance_file": "PRP.dat",
        "weight": "[1]", "alpha": 0.1, "solver_variant": strategy,
        "total_seconds": 10.0, "timed_out": timed_out,
        "first_feasible_seconds": first, "gap_target_seconds": gap,
    }


def test_writer_creates_shards_and_consolidator_generates_plotly(tmp_path):
    out = tmp_path / "out"
    write_shards(out, _summary("solver", "solver", 3.0, 8.0), [], [{"event": "first_feasible", "elapsed_seconds": 3.0}])
    write_shards(out, _summary("pso", "pso_mip_start", 1.0, 5.0), [{"iteration": 0, "best_cost": 10.0, "phase_times": {"build": 0.1}}], [])
    script = Path(__file__).parents[1] / "scripts" / "consolidate_telemetry.py"
    subprocess.run([sys.executable, str(script), "--input", str(out)], check=True)
    consolidated = out / "telemetry_consolidated"
    assert len(pd.read_parquet(consolidated / "run_summary.parquet")) == 2
    assert (consolidated / "pso_iterations.parquet").exists()
    for name in ("performance_profile_first_feasible.html", "performance_profile_gap_target.html", "performance_profile_completed_total_time.html", "success_rate_by_metric.html"):
        assert (consolidated / name).exists()


def test_writer_uses_distinct_run_files(tmp_path):
    output = tmp_path / "out"
    write_shards(output, _summary("a", "solver"), [], [])
    write_shards(output, _summary("b", "solver"), [], [])
    assert len(list((output / "telemetry").glob("*.run_summary.parquet"))) == 2
