import json
import os
from hashlib import sha1

import numpy as np
import orjson
import pandas as pd
from src.helpers.Converter import toStopPoint
from src.helpers.InstanceMetadata import enrich_with_instance_metadata


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
    guardrails=None,
    task_context=None,
):
    task_context = task_context or {}

    def _log_memory_checkpoint(checkpoint):
        if not log or not guardrails or not guardrails.is_enabled():
            return
        if not guardrails.log_memory_checkpoints:
            return
        snapshot = guardrails.snapshot()
        log.info(
            guardrails.format_snapshot(
                snapshot,
                context={
                    **task_context,
                    "event": "memory_checkpoint",
                    "phase": "write_results",
                    "checkpoint": checkpoint,
                },
            )
        )

    def _log_no_solution_results():
        if not log:
            return
        if guardrails and guardrails.is_enabled():
            snapshot = guardrails.snapshot()
            log.info(
                guardrails.format_snapshot(
                    snapshot,
                    context={
                        **task_context,
                        "event": "no_solution_results",
                        "phase": "write_results",
                    },
                )
            )
            return
        log.info("event=no_solution_results phase=write_results")

    routes = [[toStopPoint(v) for v in Z[t]] for t in range(len(Z))]

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

    s_p = np.array(data["s_p"])
    c_p = np.array(data["c_p"])
    csetup = [np.sum(s_p * Y[t]) for t in range(len(Y))]
    cprod = [np.sum(c_p * X[t]) for t in range(len(X))]

    # f1..f5 aligned with model definitions
    f1 = cprod
    f2 = csetup

    h_pi = np.array(data["h_pi"])
    I_aux = np.array(I)
    f3 = [np.sum(h_pi * I_aux[t].T) for t in range(len(I))]

    f4 = []
    f5 = []
    f = data["f"]
    a_ik = np.array(data["a_ik"])
    for t in range(len(Z)):
        sum_f5 = 0
        sum_f4 = 0
        for v in range(len(Z[t])):
            sum_f5 += np.sum(a_ik * Z[t][v])
            sum_f4 += np.sum(f * Z[t][v][0])
        f5.append(sum_f5)
        f4.append(sum_f4)

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

    _log_memory_checkpoint("before_build_df_aux")
    df_aux = pd.DataFrame(
        {
            "time": list(range(n)),
            "file": [data["file"]] * n,
            "hash_file": [file_name_hash] * n,
            "weight_hash": [weight_hash] * n,
            "weight": str(weight),
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
    _log_memory_checkpoint("before_to_excel")
    df_aux.to_excel(excel_path, index=False)
    _log_memory_checkpoint("after_to_excel")

    parquet_dir = os.path.join(dir, "parquets")
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir, f"{excel_base_name}.parquet")

    def _append_records(records, hash_file, var, idx: dict, value):
        rec = {"hash_file": hash_file, "var": var, **idx, "value": value}
        records.append(rec)

    _log_memory_checkpoint("before_build_records")
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

    _log_memory_checkpoint("before_dataframe_from_records")
    df_parquet = pd.DataFrame.from_records(records)
    _log_memory_checkpoint("after_dataframe_from_records")
    if not df_parquet.empty:
        first_cols = [c for c in ["hash_file", "var"] if c in df_parquet.columns]
        other_cols = [c for c in ["t", "p", "v", "i", "k", "j"] if c not in first_cols]
        last_cols = [c for c in ["value"] if c not in first_cols]
        df_parquet = df_parquet[first_cols + other_cols + last_cols]
    _log_memory_checkpoint("before_to_parquet")
    df_parquet.to_parquet(parquet_path, index=False)
    _log_memory_checkpoint("after_to_parquet")

    periods = []
    for t in range(len(routes)):
        veicles = []
        for v in range(len(routes[t])):
            points = []
            v_qtd_max = 0
            for i in range(len(routes[t][v])):
                products = []
                r_current = 0

                for p in range(len(Q[t][v])):
                    if i != len(routes[t][v]) - 1:
                        r_current = R[t][v][p][routes[t][v][i + 1]][routes[t][v][i]]

                    products.append(
                        {"p": p + 1, "qtd": Q[t][v][p][routes[t][v][i]], "r": r_current}
                    )
                    v_qtd_max = v_qtd_max + Q[t][v][p][routes[t][v][i]]
                points.append(
                    {
                        "point": routes[t][v][i],
                        "x": data["coordXY"]["x"][routes[t][v][i]],
                        "y": data["coordXY"]["y"][routes[t][v][i]],
                        "products": products,
                    }
                )
            veicles.append({"v": v + 1, "points": points, "v_qtd_max": v_qtd_max})

        productions = []
        for p in range(len(X[t])):
            productions.append(
                {"p": p + 1, "qtd": X[t][p], "isProduction": int(Y[t][p])}
            )

        est = []
        est_i = []
        dem = []
        for i in range(len(I[t])):
            e_p_current = []
            e_i_p_current = []
            d_p_current = []
            e_current = 0
            e_i_current = 0
            d_current = 0
            for p in range(len(I[t][i])):
                if t == 0:
                    e_i_current = data["I_pi0"][p][i]
                else:
                    e_i_current = I[t - 1][i][p]

                e_current = I[t][i][p]

                if i != len(I[t]) - 1:
                    d_current = data["d_pit"][p][i][t]

                e_i_p_current.append({"p": p + 1, "qtd": e_i_current})
                e_p_current.append({"p": p + 1, "qtd": e_current})
                d_p_current.append({"p": p + 1, "qtd": d_current})
            est_i.append({"point": i, "products": e_i_p_current})
            est.append({"point": i, "products": e_p_current})
            dem.append({"point": i + 1, "products": d_p_current})

        periods.append(
            {
                "t": t + 1,
                "veicles": veicles,
                "productions": productions,
                "dem": dem,
                "estq": est,
                "estq_i": est_i,
            }
        )

    results = {
        "hash": file_name_hash,
        "periods": periods,
        "P": P,
        "FO": FO,
        "gap": GAP,
        "time": TIME,
        "solCount": SOL_COUNT,
        "relaxeModelObjeVal": RELAXED_MODEL_OBJE_VAL,
        "nodeCount": NODE_COUNT,
        "objBound": OBJ_BOUND,
    }

    # caminho_arquivo = os.path.join(dir, "result.json")
    # with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
    #     json.dump(
    #         results,
    #         arquivo,
    #         indent=4,
    #         ensure_ascii=False,
    #     )

    return results


def new_get_results(
    dir,
    FO,
    f1,
    f2,
    f3,
    f4,
    f5,
    GAP,
    TIME,
    SOL_COUNT,
    RELAXED_MODEL_OBJE_VAL,
    NODE_COUNT,
    OBJ_BOUND,
):
    df = pd.DataFrame.from_records(
        [
            (
                dir,
                sha1(dir.encode()).hexdigest(),
                FO,
                f1,
                f2,
                f3,
                f4,
                f5,
                GAP,
                TIME,
                SOL_COUNT,
                RELAXED_MODEL_OBJE_VAL,
                NODE_COUNT,
                OBJ_BOUND,
            )
        ],
        columns=[
            "dir",
            "hash",
            "FO",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "GAP",
            "TIME",
            "SOL_COUNT",
            "RELAXED_MODEL_OBJE_VAL",
            "NODE_COUNT",
            "OBJ_BOUND",
        ],
    )
    df.to_excel(os.path.join(dir, "result.xlsx"), index=False)
