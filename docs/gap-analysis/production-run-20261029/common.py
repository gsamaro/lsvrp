"""Helpers shared by the two reports of the production run of 2026-10-29 (completo and pareado)."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

EXPERIMENTS = {
    "A": "pso_mip_polish",
    "B": "matheuristic_mip_polish",
    "C": "matheuristic",
}
LABELS = {
    "A": "PSO + MIP",
    "B": "Matheur\u00edstica + MIP",
    "C": "Matheur\u00edstica",
}
COLORS = {"A": "#1f77b4", "B": "#d95f02", "C": "#2a9d4b"}
EXPECTED_TASKS = 900


def num(value, digits=4):
    """Decimal-comma number for LaTeX text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{100 * value:.{digits}f}".replace(".", ",") + "\\%"


def weight_label(text):
    values = [float(x) for x in str(text).strip("[]").split(",")]
    return "(" + "; ".join(f"{v:g}".replace(".", ",") for v in values) + ")"


def table(columns, rows, spec=None, header_note=None):
    spec = spec or "l" + "r" * (len(columns) - 1)
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    out += [" & ".join(str(c) for c in row) + " \\\\" for row in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- loading


def load_union(input_dir, name):
    path = glob.glob(os.path.join(input_dir, name, "*-union_results.xlsx"))[0]
    frame = pd.read_excel(path)
    grouped = frame.groupby("task_key")
    tasks = grouped.first().reset_index()[
        ["task_key", "file", "weight", "alpha", "FO", "gap", "solver_time", "classe", "seeds", "instancia"]
    ]
    totals = grouped[["f1", "f2", "f3", "f4", "f5", "total_setup", "total_production"]].sum().reset_index()
    return tasks.merge(totals, on="task_key").set_index("task_key"), frame


def load_stages(input_dir):
    base = os.path.join(input_dir, EXPERIMENTS["B"], "DATA_PRP_20C")
    records = []
    for path in glob.glob(os.path.join(base, "*", "matheuristic", "*.stages.parquet")):
        metadata = json.load(open(path.replace(".stages.parquet", ".metadata.json")))
        stages = pd.read_parquet(path)
        exact = stages[stages.stage == "exact_start"].iloc[0]
        restricted = stages[stages.stage == "restricted"].iloc[0]
        instance = os.path.normpath(metadata["instance_file"])
        instance = instance[instance.index("data"):]
        records.append(
            {
                "task_key": f"{instance}|{metadata['weight']}|{metadata['alpha']:g}",
                "mip_fo": float(exact.objective),
                "mip_bound": float(exact.bound),
                "mip_gap": float(exact.gap),
                "mip_selected": bool(exact.selected),
                "restricted_fo": float(restricted.objective),
                "restricted_gap": float(restricted.gap),
                "validation_errors": ";".join(exact.validation_errors) if exact.validation_errors is not None else "",
            }
        )
    return pd.DataFrame(records).set_index("task_key")


def bootstrap_mean_ci(values, seed=0, draws=4000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


