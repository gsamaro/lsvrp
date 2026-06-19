import os
import sys
import tempfile
import types
import unittest


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


def _recursive_sum(value):
    if isinstance(value, (list, tuple)):
        return sum(_recursive_sum(item) for item in value)
    return value


fake_numpy = types.ModuleType("numpy")
fake_numpy.nan = float("nan")
fake_numpy.integer = int
fake_numpy.floating = float
fake_numpy.array = lambda value: value
fake_numpy.sum = _recursive_sum
fake_numpy.zeros = lambda shape: [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
sys.modules.setdefault("numpy", fake_numpy)

fake_orjson = types.ModuleType("orjson")
sys.modules.setdefault("orjson", fake_orjson)

fake_pandas = types.ModuleType("pandas")
fake_pandas.DataFrame = FakeDataFrame
sys.modules.setdefault("pandas", fake_pandas)

converter_module = types.ModuleType("src.helpers.Converter")
converter_module.toStopPoint = lambda matrix: matrix
sys.modules.setdefault("src.helpers.Converter", converter_module)

instance_metadata_module = types.ModuleType("src.helpers.InstanceMetadata")
instance_metadata_module.enrich_with_instance_metadata = lambda df, file_col="file": df
sys.modules.setdefault("src.helpers.InstanceMetadata", instance_metadata_module)

from src.process.ProcessResults import _build_hash_rows, getResults


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))


class DummyGuardrails:
    def is_enabled(self):
        return True

    @property
    def log_memory_checkpoints(self):
        return True

    def snapshot(self):
        return object()

    def format_snapshot(self, snapshot, context=None):
        context = context or {}
        event = context.get("event", "unknown")
        phase = context.get("phase", "unknown")
        return f"guardrail event={event} phase={phase}"


class ProcessResultsTestCase(unittest.TestCase):
    def test_build_hash_rows_returns_one_hash_per_time(self):
        hashes = _build_hash_rows("abc123", [0, 1, 2])

        self.assertEqual(len(hashes), 3)
        self.assertEqual(len(set(hashes)), 3)
        self.assertTrue(all(isinstance(item, str) and len(item) == 40 for item in hashes))

    def test_get_results_without_solution_does_not_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DummyLogger()
            guardrails = DummyGuardrails()
            data = {
                "file": "./data/DATA_PRP_30C/PRP22_C30_P10_V5_T12_S2.dat",
                "weight": [0.2, 0.2, 0.2, 0.2, 0.2],
                "alpha": 0.01,
                "s_p": [1.0],
                "c_p": [2.0],
                "h_pi": [[0.0]],
                "f": [[0.0]],
                "a_ik": [[0.0]],
                "num_periods": 2,
                "coordXY": {"x": [0.0], "y": [0.0]},
                "I_pi0": [[0.0]],
                "d_pit": [[[0.0, 0.0]]],
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
                guardrails=guardrails,
                task_context={"mpi_batch": 1, "task_number": 1},
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
            guardrails = DummyGuardrails()
            data = {
                "file": "./data/DATA_PRP_30C/PRP22_C30_P10_V5_T12_S2.dat",
                "weight": [0.2, 0.2, 0.2, 0.2, 0.2],
                "alpha": 0.01,
                "s_p": [1.0],
                "c_p": [2.0],
                "h_pi": [[0.0]],
                "f": [[0.0]],
                "a_ik": [[0.0]],
                "num_periods": 1,
                "coordXY": {"x": [0.0], "y": [0.0]},
                "I_pi0": [[0.0]],
                "d_pit": [[[0.0]]],
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
                guardrails=guardrails,
                task_context={"mpi_batch": 1, "task_number": 1},
            )

            parquet_dir = os.path.join(tmpdir, "parquets")
            parquet_files = os.listdir(parquet_dir)
            self.assertEqual(results["periods"], [])
            self.assertEqual(len(parquet_files), 1)
            with open(os.path.join(parquet_dir, parquet_files[0]), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "hash_file,var,t,j,value")


if __name__ == "__main__":
    unittest.main()
