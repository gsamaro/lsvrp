from pathlib import Path
import pandas as pd
from typing import Any
from constants import FILE

EXCLUDED_WEIGHTS = [
    "[1, 0, 0, 0, 0]",
    "[0, 1, 0, 0, 0]",
    "[0, 0, 1, 0, 0]",
    "[0, 0, 0, 1, 0]",
    "[0, 0, 0, 0, 1]",
]


def read_results_df() -> pd.DataFrame:
    path = Path().absolute() / "out" / FILE
    df = pd.read_excel(path, engine="openpyxl")
    df = df[~df["weight"].isin(EXCLUDED_WEIGHTS)].copy()
    df = df[~df["instancia"].isna()].copy()
    try:
        df["instancia"] = df["instancia"].astype(int)
    except Exception:
        pass
    
    #TODO: Remover de ter todos os resultados completos.
    df = df.query("clientes <= 20")

    # map weight_hash -> label
    hash_list = df["weight_hash"].unique()

    def make_weight_hash_map_from_list(hash_list, start_at=1):
        return {h: rf"$w_{{{i}}}$" for i, h in enumerate(hash_list, start=start_at)}

    hash_map = make_weight_hash_map_from_list(hash_list)
    df["weight_label"] = df["weight_hash"].map(hash_map).fillna(df["weight_hash"])

    # compute relative deviations for FOBs when source columns exist
    for i in range(1, 6):
        p_col = f"p{i}"
        target_col = f"f{i}_target"
        out_col = f"desv_rel_targ_{i}"
        if p_col in df.columns and target_col in df.columns:
            # avoid division by zero
            df[out_col] = df[p_col] / df[target_col].replace({0: pd.NA})

    # compute f_sum_t if not present
    if "f_sum_t" not in df.columns:
        f_cols = [c for c in ["f1", "f2", "f3", "f4", "f5"] if c in df.columns]
        if f_cols:
            df["f_sum_t"] = df[f_cols].sum(axis=1)

    return df
