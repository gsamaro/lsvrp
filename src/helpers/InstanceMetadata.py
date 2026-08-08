import os
import re

import pandas as pd

_PATTERN = re.compile(r"PRP(\d+)_C(\d+)_P(\d+)_V(\d+)_T(\d+)_S(\d+)")


def enrich_with_instance_metadata(
    df: pd.DataFrame, file_col: str = "file"
) -> pd.DataFrame:
    if df is None:
        raise ValueError("df cannot be None")
    if file_col not in df.columns:
        return df.copy()

    out = df.copy()

    file_series = out[file_col].astype("string")

    def _normalize_file(value):
        if value is None or pd.isna(value):
            return pd.NA
        base = os.path.basename(str(value))
        stem, _ = os.path.splitext(base)
        return stem

    normalized = file_series.map(_normalize_file)

    extracted = normalized.str.extract(
        _PATTERN,
        expand=True,
    )
    extracted.columns = [
        "instancia",
        "clientes",
        "produtos",
        "veiculos",
        "periodos",
        "seeds",
    ]

    out = pd.concat([out, extracted], axis=1)

    for c in ["instancia", "clientes", "produtos", "veiculos", "periodos", "seeds"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    bins = [0, 10, 20, 30, 40]
    labels = [1, 2, 3, 4]
    out["classe"] = (
        pd.cut(out["instancia"], bins=bins, labels=labels, include_lowest=True)
        .astype("Int64")
        .copy()
    )

    return out
