import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.reports.sensitivity import _get_ct, generate_sensitivity_tables


class SensitivityReportTestCase(unittest.TestCase):
    def test_get_ct_uses_file_and_weight_hash_as_stable_key(self):
        df = pd.DataFrame(
            [
                {
                    "file": "instance_a.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.01,
                    "clientes": 5,
                    "classe": 1,
                    "f_sum_t": 10.0,
                },
                {
                    "file": "instance_a.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.99,
                    "clientes": 5,
                    "classe": 1,
                    "f_sum_t": 12.0,
                },
                {
                    "file": "instance_a.dat",
                    "weight_hash": "zzz999",
                    "weight_label": "$w_{2}$",
                    "alpha": 0.01,
                    "clientes": 5,
                    "classe": 1,
                    "f_sum_t": 20.0,
                },
                {
                    "file": "instance_a.dat",
                    "weight_hash": "zzz999",
                    "weight_label": "$w_{2}$",
                    "alpha": 0.99,
                    "clientes": 5,
                    "classe": 1,
                    "f_sum_t": 10.0,
                },
                {
                    "file": "instance_b.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.01,
                    "clientes": 10,
                    "classe": 2,
                    "f_sum_t": 7.0,
                },
                {
                    "file": "instance_b.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.99,
                    "clientes": 10,
                    "classe": 2,
                    "f_sum_t": 14.0,
                },
            ]
        )

        result = _get_ct(df, "clientes")

        self.assertEqual(list(result.columns), ["weight_label", "clientes", "ct"])
        self.assertEqual(len(result), 3)

        row_w1 = result[result["weight_label"] == "$w_{1}$"].iloc[0]
        row_w2 = result[result["weight_label"] == "$w_{2}$"].iloc[0]
        self.assertAlmostEqual(row_w1["ct"], 0.2)
        self.assertAlmostEqual(row_w2["ct"], 0.5)

    def test_get_ct_returns_empty_dataframe_when_no_alpha_pair_matches(self):
        df = pd.DataFrame(
            [
                {
                    "file": "instance_a.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.01,
                    "clientes": 5,
                    "classe": 1,
                    "f_sum_t": 10.0,
                },
                {
                    "file": "instance_a.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.99,
                    "clientes": 10,
                    "classe": 1,
                    "f_sum_t": 12.0,
                },
            ]
        )

        result = _get_ct(df, "clientes")

        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["weight_label", "clientes", "ct"])

    def test_generate_sensitivity_tables_skips_empty_ct_tables(self):
        fake_df = pd.DataFrame(
            [
                {
                    "file": "instance_a.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.01,
                    "clientes": 5,
                    "classe": 1,
                    "f1": 1.0,
                    "f2": 1.0,
                    "f3": 1.0,
                    "f4": 1.0,
                    "f5": 1.0,
                },
                {
                    "file": "instance_a.dat",
                    "weight_hash": "abc123",
                    "weight_label": "$w_{1}$",
                    "alpha": 0.99,
                    "clientes": 10,
                    "classe": 2,
                    "f1": 2.0,
                    "f2": 2.0,
                    "f3": 2.0,
                    "f4": 2.0,
                    "f5": 2.0,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("src.reports.sensitivity.read_results_df", return_value=fake_df):
                generate_sensitivity_tables(out_dir)

            clientes_tex = (out_dir / "latex" / "sensitivity_ct_clientes.tex").read_text()
            classe_tex = (out_dir / "latex" / "sensitivity_ct_classe.tex").read_text()

            self.assertIn("Sensitivity table skipped", clientes_tex)
            self.assertIn("Sensitivity table skipped", classe_tex)


if __name__ == "__main__":
    unittest.main()
