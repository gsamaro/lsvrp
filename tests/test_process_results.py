from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd

from src.process.ProcessResults import _build_hash_rows, getResults


EXCEL_COLUMNS = [
    "time", "file", "hash_file", "weight_hash", "weight", "alpha", "FO", "gap",
    "solver_time", "f1", "f2", "f3", "f4", "f5", "p1", "p2", "p3", "p4",
    "p5", "epsilon", "total_production", "total_inventory", "total_setup",
    "total_delivered", "csetup", "cprod", "instancia", "clientes", "produtos",
    "veiculos", "periodos", "seeds", "classe", "new_f1_target", "new_f2_target",
    "new_f3_target", "new_f4_target", "new_f5_target", "hash_row",
]


class DummyLogger:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        self.messages.append(message)


def _data(num_periods=1):
    return {
        "file": "./data/DATA_PRP_30C/PRP22_C30_P10_V5_T12_S2.dat",
        "weight": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        "alpha": 0.01,
        "s_p": np.array([1.0]),
        "c_p": np.array([2.0]),
        "h_pi": np.array([[0.0, 0.0]]),
        "f": np.array([[0.0, 0.0], [0.0, 0.0]]),
        "a_ik": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "num_periods": num_periods,
    }


def _write_results(output, data, Z, X, Y, I, R, Q, P, logger=None):
    return getResults(
        data, output, Z, X, Y, I, R, Q, P, 10.0, 0.1, 12.5, None, 1,
        0.0, 0, 0.0, None, log=logger,
    )


def _generated_paths(output):
    excel_paths = list(Path(output).glob("*_fobs.xlsx"))
    parquet_paths = list((Path(output) / "parquets").glob("*_fobs.parquet"))
    assert len(excel_paths) == 1
    assert len(parquet_paths) == 1
    return excel_paths[0], parquet_paths[0]


def test_build_hash_rows_returns_one_hash_per_time():
    hashes = _build_hash_rows("abc123", [0, 1, 2])

    assert len(hashes) == 3
    assert len(set(hashes)) == 3
    assert all(isinstance(item, str) and len(item) == 40 for item in hashes)


def test_get_results_writes_empty_solution_artifacts_and_returns_none(tmp_path):
    logger = DummyLogger()

    result = _write_results(
        tmp_path, _data(num_periods=2), [], [], [], [], [], [], [], logger
    )

    assert result is None
    excel_path, parquet_path = _generated_paths(tmp_path)
    excel_df = pd.read_excel(excel_path, engine="openpyxl")
    parquet_df = pd.read_parquet(parquet_path)
    assert list(excel_df.columns) == EXCEL_COLUMNS
    assert excel_df.empty
    assert list(parquet_df.columns) == ["hash_file", "var", "t", "j", "value"]
    assert len(parquet_df) == 10
    assert set(parquet_df["var"]) == {"P"}
    assert "event=no_solution_results phase=write_results" in logger.messages


def test_get_results_preserves_excel_and_parquet_contract(tmp_path):
    data = _data()
    Z = [[[[0.0, 1.0], [0.0, 0.0]]]]
    X = [[1.0]]
    Y = [[0.0]]
    I = [[[0.0], [0.0]]]
    R = [[[[[0.0, 0.0], [0.0, 0.0]]]]]
    Q = [[[[0.0, 0.0]]]]
    P = [[0.0, 0.0, 0.0, 0.0, 0.0]]

    result = _write_results(tmp_path, data, Z, X, Y, I, R, Q, P)

    assert result is None
    excel_path, parquet_path = _generated_paths(tmp_path)
    excel_df = pd.read_excel(excel_path, engine="openpyxl")
    parquet_df = pd.read_parquet(parquet_path)
    file_hash = sha1((data["file"] + str(data["weight"]) + str(data["alpha"])).encode()).hexdigest()

    assert list(excel_df.columns) == EXCEL_COLUMNS
    assert excel_df.loc[0, "hash_file"] == file_hash
    assert excel_df.loc[0, "hash_row"] == _build_hash_rows(file_hash, [0])[0]
    weight_payload = ",".join("{:.10g}".format(float(value)) for value in data["weight"])
    assert excel_df.loc[0, "weight_hash"] == sha1(weight_payload.encode()).hexdigest()[:6]
    assert excel_df.loc[0, ["f1", "f2", "f3", "f4", "f5"]].tolist() == [2, 0, 0, 0, 1]

    assert list(parquet_df.columns) == [
        "hash_file", "var", "t", "p", "v", "i", "k", "j", "value",
    ]
    assert len(parquet_df) == 19
    assert set(parquet_df["var"]) == {"Z", "X", "Y", "I", "R", "Q", "P"}
