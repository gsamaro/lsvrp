import json
import os
from src.helpers.Converter import toStopPoint
import pandas as pd
from hashlib import sha1
import numpy as np
import orjson


def getResults(
    data,
    dir,
    Z,
    X,
    Y,
    I,
    R,
    Q,
    FO,
    GAP,
    TIME,
    SOL_COUNT,
    RELAXED_MODEL_OBJE_VAL,
    NODE_COUNT,
    OBJ_BOUND,
):
    routes = [[toStopPoint(v) for v in Z[t]] for t in range(len(Z))]

    weight = data['weight']

    s_p = np.array(data["s_p"])
    c_p = np.array(data["c_p"])
    csetup = [np.sum(s_p * Y[t]) for t in range(len(Y))]
    cprod = [np.sum(c_p * X[t]) for t in range(len(X))]

    # Keep as a plain Python list (avoid ndarray for JSON serialization)
    f1 = [cs + cp for cs, cp in zip(csetup, cprod)]

    h_pi = np.array(data["h_pi"])
    I_aux = np.array(I)
    f2 = [np.sum(h_pi * I_aux[t].T) for t in range(len(I))]

    f3 = []
    f4 = []
    f = data["f"]
    a_ik = np.array(data["a_ik"])
    for t in range(len(Z)):
        sum_f4 = 0
        sum_f3 = 0
        for v in range(len(Z[t])):
            sum_f4 += np.sum(a_ik * Z[t][v])
            sum_f3 += np.sum(f * Z[t][v][0])
        f4.append(sum_f4)
        f3.append(sum_f3)

    file_name_hash = sha1((data['file']+str(weight)).encode()).hexdigest()
    # Build one row per period with consistent native types
    n = len(f1)
    df_aux = pd.DataFrame({
        "time": list(range(n)),
        "file": [data['file']] * n,
        "hash_file": [file_name_hash] * n,
        "csetup": [float(x) for x in csetup],
        "cprod": [float(x) for x in cprod],
        "f1": [float(x) for x in f1],
        "f2": [float(x) for x in f2],
        "f3": [float(x) for x in f3],
        "f4": [float(x) for x in f4],
        "weight": str(weight)
    })
    # Generate a unique SHA1 per row based on stable string representation
    def _row_hash(row):
        payload = (
            f"{file_name_hash}|{row['time']}|"
        )
        return sha1(payload.encode('utf-8')).hexdigest()
    df_aux["hash_row"] = df_aux.apply(_row_hash, axis=1)
    df_aux.to_excel(os.path.join(dir, f"{file_name_hash[:6]}_fobs.xlsx"), index=False)

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
        "FO": FO,
        "gap": GAP,
        "time": TIME,
        "solCount": SOL_COUNT,
        "relaxeModelObjeVal": RELAXED_MODEL_OBJE_VAL,
        "nodeCount": NODE_COUNT,
        "objBound": OBJ_BOUND,
    }

    caminho_arquivo = os.path.join(dir, "result.json")
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(
            results,
            arquivo,
            indent=4,
            ensure_ascii=False,
        )

    return results


def new_get_results(
    dir,
    FO,
    f1,
    f2,
    f3,
    f4,
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
            "GAP",
            "TIME",
            "SOL_COUNT",
            "RELAXED_MODEL_OBJE_VAL",
            "NODE_COUNT",
            "OBJ_BOUND",
        ],
    )
    df.to_excel(os.path.join(dir, "result.xlsx"), index=False)
