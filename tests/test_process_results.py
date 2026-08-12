import os
import sys
import tempfile
import types
import unittest

import numpy as np


class FakeSeries(list):
    def tolist(self):
        return list(self)


class FakeDataFrame:
    def __init__(self, data=None):
        self._data = {}
        self.columns = []
        if isinstance(data, dict):
            for key, value in data.items():
                self._data[key] = list(value) if isinstance(value, list) else value
            self.columns = list(data.keys())
        elif data is None:
            self._data = {}
            self.columns = []

    @property
    def empty(self):
        if not self.columns:
            return True
        first_column = self.columns[0]
        values = self._data.get(first_column, [])
        return len(values) == 0 if isinstance(values, list) else False

    def __getitem__(self, key):
        if isinstance(key, list):
            missing = [column for column in key if column not in self.columns]
            if missing:
                raise KeyError(f"{missing} not in index")
            subset = {column: self._data.get(column, []) for column in key}
            return FakeDataFrame(subset)
        return FakeSeries(self._data.get(key, []))

    def __setitem__(self, key, value):
        self._data[key] = list(value) if isinstance(value, (list, tuple, FakeSeries)) else value
        if key not in self.columns:
            self.columns.append(key)

    def to_excel(self, path, index=False):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(",".join(self.columns))

    def to_parquet(self, path, index=False):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(",".join(self.columns))

    @classmethod
    def from_records(cls, records):
        if not records:
            return cls({})
        columns = list(records[0].keys())
        data = {column: [record.get(column) for record in records] for column in columns}
        return cls(data)


fake_pandas = types.ModuleType("pandas")
fake_pandas.DataFrame = FakeDataFrame
sys.modules.setdefault("pandas", fake_pandas)

instance_metadata_module = types.ModuleType("src.helpers.InstanceMetadata")
instance_metadata_module.enrich_with_instance_metadata = lambda df, file_col="file": df
sys.modules.setdefault("src.helpers.InstanceMetadata", instance_metadata_module)

from src.process.ProcessResults import _build_hash_rows, _build_routes_from_Z, getResults


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def debug(self, message):
        self.messages.append(("debug", message))


class ProcessResultsTestCase(unittest.TestCase):
    def test_build_routes_from_z_reconstructs_sequence(self):
        z = [
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ]

        self.assertEqual(_build_routes_from_Z(z), [[[0, 1, 2]]])

    def test_build_routes_from_z_handles_empty_vehicle(self):
        z = [[[ [0.0, 0.0], [0.0, 0.0] ]]]

        self.assertEqual(_build_routes_from_Z(z), [[[]]])

    def test_build_routes_from_z_stops_on_incomplete_cycle(self):
        z = [
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            ]
        ]

        self.assertEqual(_build_routes_from_Z(z), [[[0, 1, 2, 0]]])

    def test_build_hash_rows_returns_one_hash_per_time(self):
        hashes = _build_hash_rows("abc123", [0, 1, 2])

        self.assertEqual(len(hashes), 3)
        self.assertEqual(len(set(hashes)), 3)
        self.assertTrue(all(isinstance(item, str) and len(item) == 40 for item in hashes))

    def test_get_results_without_solution_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DummyLogger()
            data = {
                "file": "./data/DATA_PRP_30C/PRP22_C30_P10_V5_T12_S2.dat",
                "weight": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                "alpha": 0.01,
                "s_p": np.array([1.0]),
                "c_p": np.array([2.0]),
                "h_pi": np.array([[0.0]]),
                "f": np.array([[0.0]]),
                "a_ik": np.array([[0.0]]),
                "num_periods": 2,
                "coordXY": {"x": [0.0], "y": [0.0]},
                "I_pi0": np.array([[0.0]]),
                "d_pit": np.array([[[0.0, 0.0]]]),
            }

            results = getResults(
                data,
                tmpdir,
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                0,
                0,
                12.5,
                None,
                0,
                0,
                0,
                0,
                None,
                log=logger,
            )

            generated_files = os.listdir(tmpdir)
            self.assertEqual(results["periods"], [])
            self.assertTrue(any(name.endswith(".xlsx") for name in generated_files))
            self.assertIn("parquets", generated_files)
            self.assertTrue(
                any("event=no_solution_results" in message for _, message in logger.messages)
            )

    def test_get_results_writes_parquet_even_when_some_index_columns_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DummyLogger()
            data = {
                "file": "./data/DATA_PRP_30C/PRP22_C30_P10_V5_T12_S2.dat",
                "weight": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                "alpha": 0.01,
                "s_p": np.array([1.0]),
                "c_p": np.array([2.0]),
                "h_pi": np.array([[0.0]]),
                "f": np.array([[0.0]]),
                "a_ik": np.array([[0.0]]),
                "num_periods": 1,
                "coordXY": {"x": [0.0], "y": [0.0]},
                "I_pi0": np.array([[0.0]]),
                "d_pit": np.array([[[0.0]]]),
            }

            results = getResults(
                data,
                tmpdir,
                [],
                [],
                [],
                [],
                [],
                [],
                [[0.0]],
                0,
                0,
                12.5,
                None,
                0,
                0,
                0,
                0,
                None,
                log=logger,
            )

            parquet_dir = os.path.join(tmpdir, "parquets")
            parquet_files = os.listdir(parquet_dir)
            self.assertEqual(results["periods"], [])
            self.assertEqual(len(parquet_files), 1)
            parquet_path = os.path.join(parquet_dir, parquet_files[0])
            import pandas as pd

            if hasattr(pd, "read_parquet"):
                parquet_df = pd.read_parquet(parquet_path)
                self.assertEqual(list(parquet_df.columns), ["hash_file", "var", "t", "j", "value"])
            else:
                with open(parquet_path, encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "hash_file,var,t,j,value")

    def test_get_results_reconstructs_routes_without_converter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DummyLogger()
            data = {
                "file": "./data/DATA_PRP_30C/PRP22_C30_P10_V5_T12_S2.dat",
                "weight": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                "alpha": 0.01,
                "s_p": np.array([1.0]),
                "c_p": np.array([2.0]),
                "h_pi": np.array([[0.0]]),
                "f": np.array([[0.0, 0.0], [0.0, 0.0]]),
                "a_ik": np.array([[0.0, 1.0], [1.0, 0.0]]),
                "num_periods": 1,
                "num_customers": 1,
                "coordXY": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "I_pi0": np.array([[0.0], [0.0]]),
                "d_pit": np.array([[[0.0]], [[0.0]]]),
            }
            z = [
                [
                    [
                        [0.0, 1.0],
                        [0.0, 0.0],
                    ]
                ]
            ]

            results = getResults(
                data,
                tmpdir,
                z,
                [[1.0]],
                [[0.0]],
                [[[0.0, 0.0]]],
                [[[[[0.0, 0.0], [0.0, 0.0]]]]],
                [[[[0.0, 0.0]]]],
                [[0.0, 0.0, 0.0, 0.0, 0.0]],
                0,
                0,
                12.5,
                None,
                0,
                0,
                0,
                0,
                None,
                log=logger,
            )

            self.assertEqual(results["periods"][0]["veicles"][0]["points"][0]["point"], 0)
            self.assertEqual(results["periods"][0]["veicles"][0]["points"][1]["point"], 1)


if __name__ == "__main__":
    unittest.main()
