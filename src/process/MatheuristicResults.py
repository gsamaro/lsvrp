"""MPI-safe complementary stage outputs; no plots or offline analyses here."""

import json
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.helpers.SolverTelemetry import _json_value, new_run_id

INDICES = {
    "u": ("i", "k"),
    "eta": ("v", "i", "t"),
    "X": ("p", "t"),
    "Y": ("p", "t"),
    "I": ("p", "i", "t"),
    "Q": ("p", "v", "i", "t"),
    "R": ("p", "v", "i", "k", "t"),
    "Z": ("v", "i", "k", "t"),
    "P": ("j", "t"),
    "N": ("j", "t"),
    "lambda": (),
}


def write_artifacts(output_dir, data, artifacts, task_context=None):
    directory = Path(output_dir) / "matheuristic"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(task_context or {}, "matheuristic")
    hash_file = sha1(
        (data["file"] + str(data["weight"]) + str(data["alpha"])).encode()
    ).hexdigest()
    metadata = {
        **artifacts["metadata"],
        "run_id": run_id,
        "hash_file": hash_file,
        "schema_version": 1,
        "index_origin": 0,
        "available_variables": {},
    }
    final_path = Path(output_dir) / "parquets" / f"{hash_file[:6]}_fobs.parquet"
    if final_path.exists():
        stat = final_path.stat()
        metadata["final_output"] = {
            "name": final_path.name,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    schema = pa.schema(
        [(key, pa.string()) for key in ("run_id", "hash_file", "stage", "var")]
        + [(key, pa.int64()) for key in ("t", "p", "v", "i", "k", "j")]
        + [("value", pa.float64())]
    )
    batch = []
    with pq.ParquetWriter(
        directory / f"{run_id}.variables.parquet", schema, compression="zstd"
    ) as writer:

        def flush():
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()

        for artifact in artifacts["variables"]:
            stage = artifact["stage"]
            metadata["available_variables"][stage] = {}
            for var, values in artifact["variables"].items():
                sparse = var not in ("u", "eta")
                if isinstance(values, dict):
                    dimensions = metadata.get("dimensions") or {
                        "p": data["num_products"],
                        "v": data["num_vehicles"],
                        "i": data["num_customers"] + 1,
                        "t": data["num_periods"],
                    }
                    shape = tuple(
                        dimensions["i" if key == "k" else key] for key in INDICES[var]
                    )
                    rows = ((key, value) for key, value in values.items() if value != 0)
                else:
                    values = np.asarray(values)
                    shape = values.shape
                    coordinates = (
                        np.argwhere(values != 0) if sparse else np.ndindex(shape)
                    )
                    rows = ((tuple(key), values[tuple(key)]) for key in coordinates)
                metadata["available_variables"][stage][var] = {
                    "shape": shape,
                    "indices": INDICES[var],
                    "sparse": sparse,
                }
                for coordinate, value in rows:
                    coordinate = tuple(coordinate)
                    batch.append(
                        {
                            "run_id": run_id,
                            "hash_file": hash_file,
                            "stage": stage,
                            "var": var,
                            **dict(zip(INDICES[var], coordinate)),
                            "value": float(value),
                        }
                    )
                    if len(batch) >= 4096:
                        flush()
        flush()
    summaries = [
        {"run_id": run_id, "hash_file": hash_file, **stage}
        for stage in artifacts["stages"]
    ]
    pd.DataFrame(summaries).to_parquet(
        directory / f"{run_id}.stages.parquet", index=False
    )
    with (directory / f"{run_id}.metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_value(metadata), stream, ensure_ascii=False, indent=2)
    return run_id
