"""Persistência leve de telemetria de execuções Solver e PSO."""
import json
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def get_config(config):
    configured = config.get_nested("solver", "telemetry", default={}) or {}
    return {
        "enabled": bool(configured.get("enabled", False)),
        "experiment_id": configured.get("experiment_id"),
        "gap_target_relative": float(configured.get("gap_target_relative", 0.01)),
        "save_pso_iterations": bool(configured.get("save_pso_iterations", True)),
    }


def new_run_id(task_context, strategy):
    payload = "|".join(str(task_context.get(k, "")) for k in ("run_tag", "task_number", "file", "weight", "alpha"))
    return f"{sha1(payload.encode()).hexdigest()[:10]}-{strategy}-{uuid4().hex[:10]}"


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _flatten_iteration(metrics):
    row = {key: value for key, value in metrics.items() if key != "phase_times"}
    for key, value in (metrics.get("phase_times") or {}).items():
        row[f"phase_{key}_seconds"] = value
    return _json_value(row)


def write_shards(output_dir, summary, pso_iterations, mip_events):
    """Escreve shards exclusivos por execução, seguros em MPI."""
    telemetry_dir = Path(output_dir) / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    run_id = summary["run_id"]
    summary = {"schema_version": SCHEMA_VERSION, **_json_value(summary)}
    pd.DataFrame([summary]).to_parquet(telemetry_dir / f"{run_id}.run_summary.parquet", index=False)
    if pso_iterations:
        rows = [{"run_id": run_id, **_flatten_iteration(item)} for item in pso_iterations]
        pd.DataFrame(rows).to_parquet(telemetry_dir / f"{run_id}.pso_iterations.parquet", index=False)
    if mip_events:
        rows = [{"run_id": run_id, **_json_value(item)} for item in mip_events]
        pd.DataFrame(rows).to_parquet(telemetry_dir / f"{run_id}.mip_events.parquet", index=False)
    with (telemetry_dir / f"{run_id}.metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "schema_version": SCHEMA_VERSION}, stream)
    return telemetry_dir
