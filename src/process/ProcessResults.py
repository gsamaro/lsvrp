import os
from hashlib import sha1
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.helpers.InstanceMetadata import enrich_with_instance_metadata
from src.solvers._solver_common import solution_components


def _build_hash_rows(file_name_hash, times):
    return [
        sha1(f"{file_name_hash}|{time_value}|".encode("utf-8")).hexdigest()
        for time_value in times
    ]


def getResults(
    data,
    dir,
    Z,
    X,
    Y,
    I,
    R,
    Q,
    P,
    FO,
    GAP,
    TIME,
    EPSILON,
    SOL_COUNT,
    RELAXED_MODEL_OBJE_VAL,
    NODE_COUNT,
    OBJ_BOUND,
    NEW_TARGETS,
    log=None,
):
    def _log_no_solution_results():
        if not log:
            return
        log.debug("event=no_solution_results phase=write_results")

    weight = data["weight"]

    try:
        weight_payload = ",".join(
            [
                (
                    ("{:.10g}".format(float(w)))
                    if (isinstance(w, (int, float, np.integer, np.floating)))
                    else str(w)
                )
                for w in weight
            ]
        )
    except Exception:
        weight_payload = str(weight)
    weight_hash = sha1(weight_payload.encode("utf-8")).hexdigest()[:6]

    if len(X):
        cost_data = SimpleNamespace(
            t=len(X),
            c_p=data["c_p"],
            s_p=data["s_p"],
            h_p_i=data["h_pi"],
            f=data["f"],
            a_i_k=data["a_ik"],
        )
        variables = {
            "X": np.asarray(X, dtype=float).T,
            "Y": np.asarray(Y, dtype=float).T,
            "I": np.asarray(I, dtype=float).transpose(2, 1, 0),
            "Z": np.asarray(Z, dtype=float).transpose(1, 2, 3, 0),
        }
        f1, f2, f3, f4, f5 = solution_components(cost_data, variables).tolist()
    else:
        f1, f2, f3, f4, f5 = [], [], [], [], []
    cprod, csetup = f1, f2

    file_name_hash = sha1(
        (data["file"] + str(weight) + str(data["alpha"])).encode()
    ).hexdigest()
    # Build one row per period with consistent native types
    n = len(f1)
    if n == 0:
        _log_no_solution_results()

    p_cols = {}
    for j in range(5):
        col = f"p{j+1}"
        if P and len(P) == n and all(len(P[t]) >= 5 for t in range(n)):
            p_cols[col] = [float(P[t][j]) for t in range(n)]
        else:
            p_cols[col] = [np.nan] * n

    df_aux = pd.DataFrame(
        {
            "time": list(range(n)),
            "file": [data["file"]] * n,
            "commit_hash": [data.get("commit_hash")] * n,
            "config_hash": [data.get("config_hash")] * n,
            "hash_file": [file_name_hash] * n,
            "weight_hash": [weight_hash] * n,
            "weight": str(weight),
            "alpha": [float(data["alpha"])] * n,
            "FO": [float(FO)] * n,
            "gap": [float(GAP)] * n,
            "solver_time": [float(TIME)] * n,
            "f1": [float(x) for x in f1],
            "f2": [float(x) for x in f2],
            "f3": [float(x) for x in f3],
            "f4": [float(x) for x in f4],
            "f5": [float(x) for x in f5],
            **p_cols,
            "epsilon": [float(EPSILON) if EPSILON is not None else np.nan] * n,
            "total_production": [float(np.sum(X))] * n,
            "total_inventory": [float(np.sum(I))] * n,
            "total_setup": [float(np.sum(Y))] * n,
            "total_delivered": [float(np.sum(Q))] * n,
            "csetup": [float(x) for x in csetup],
            "cprod": [float(x) for x in cprod],
        }
    )

    df_aux = enrich_with_instance_metadata(df_aux, file_col="file")

    def _get_new_target(t, key):
        if not NEW_TARGETS:
            return np.nan
        try:
            return float(NEW_TARGETS[t][key])
        except Exception:
            return np.nan

    df_aux["new_f1_target"] = [_get_new_target(t, "f1_target") for t in range(n)]
    df_aux["new_f2_target"] = [_get_new_target(t, "f2_target") for t in range(n)]
    df_aux["new_f3_target"] = [_get_new_target(t, "f3_target") for t in range(n)]
    df_aux["new_f4_target"] = [_get_new_target(t, "f4_target") for t in range(n)]
    df_aux["new_f5_target"] = [_get_new_target(t, "f5_target") for t in range(n)]

    df_aux["hash_row"] = _build_hash_rows(file_name_hash, list(range(n)))

    excel_base_name = f"{file_name_hash[:6]}_fobs"
    excel_path = os.path.join(dir, f"{excel_base_name}.xlsx")
    config_records = []
    if data.get("config_hash") and data.get("config_json"):
        config_records.append(
            {
                "config_hash": data["config_hash"],
                "config_json": data["config_json"],
            }
        )
    run_configs_df = pd.DataFrame(
        config_records, columns=["config_hash", "config_json"]
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_aux.to_excel(writer, index=False)
        run_configs_df.to_excel(writer, sheet_name="run_configs", index=False)

    parquet_dir = os.path.join(dir, "parquets")
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir, f"{excel_base_name}.parquet")

    def _append_records(records, hash_file, var, idx: dict, value):
        rec = {"hash_file": hash_file, "var": var, **idx, "value": value}
        records.append(rec)

    records = []

    # Z[t][v][i][k]
    for t in range(len(Z)):
        for v in range(len(Z[t])):
            for i_idx in range(len(Z[t][v])):
                for k_idx in range(len(Z[t][v][i_idx])):
                    _append_records(
                        records,
                        file_name_hash,
                        "Z",
                        {"t": t, "v": v, "i": i_idx, "k": k_idx},
                        float(Z[t][v][i_idx][k_idx]),
                    )

    # X[t][p]
    for t in range(len(X)):
        for p_idx in range(len(X[t])):
            _append_records(
                records,
                file_name_hash,
                "X",
                {"t": t, "p": p_idx},
                float(X[t][p_idx]),
            )

    # Y[t][p]
    for t in range(len(Y)):
        for p_idx in range(len(Y[t])):
            _append_records(
                records,
                file_name_hash,
                "Y",
                {"t": t, "p": p_idx},
                float(Y[t][p_idx]),
            )

    # I[t][i][p] in ProcessResults (note: computed as list over i then p)
    for t in range(len(I)):
        for i_idx in range(len(I[t])):
            for p_idx in range(len(I[t][i_idx])):
                _append_records(
                    records,
                    file_name_hash,
                    "I",
                    {"t": t, "p": p_idx, "i": i_idx},
                    float(I[t][i_idx][p_idx]),
                )

    # R[t][v][p][i][k]
    for t in range(len(R)):
        for v in range(len(R[t])):
            for p_idx in range(len(R[t][v])):
                for i_idx in range(len(R[t][v][p_idx])):
                    for k_idx in range(len(R[t][v][p_idx][i_idx])):
                        _append_records(
                            records,
                            file_name_hash,
                            "R",
                            {"t": t, "p": p_idx, "v": v, "i": i_idx, "k": k_idx},
                            float(R[t][v][p_idx][i_idx][k_idx]),
                        )

    # Q[t][v][p][i]
    for t in range(len(Q)):
        for v in range(len(Q[t])):
            for p_idx in range(len(Q[t][v])):
                for i_idx in range(len(Q[t][v][p_idx])):
                    _append_records(
                        records,
                        file_name_hash,
                        "Q",
                        {"t": t, "p": p_idx, "v": v, "i": i_idx},
                        float(Q[t][v][p_idx][i_idx]),
                    )

    # P[t][j]
    if not P:
        P = np.zeros((data["num_periods"], 5))
    for t in range(len(P)):
        for j in range(len(P[t])):
            _append_records(
                records,
                file_name_hash,
                "P",
                {"t": t, "j": j},
                float(P[t][j]),
            )

    df_parquet = pd.DataFrame.from_records(records)
    if not df_parquet.empty:
        first_cols = [c for c in ["hash_file", "var"] if c in df_parquet.columns]
        other_cols = [
            c
            for c in ["t", "p", "v", "i", "k", "j"]
            if c in df_parquet.columns and c not in first_cols
        ]
        last_cols = [
            c for c in ["value"] if c in df_parquet.columns and c not in first_cols
        ]
        df_parquet = df_parquet[first_cols + other_cols + last_cols]
    df_parquet.to_parquet(parquet_path, index=False)
