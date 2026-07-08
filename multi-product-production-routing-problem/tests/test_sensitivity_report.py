import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.reports.sensitivity import _get_ct, generate_sensitivity_tables
from src.reports.sensitivity import df_to_latex_table


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

    def test_df_to_latex_table_places_font_size_before_tabular_and_caption_after(self):
        df = pd.DataFrame(
            [
                {"group": "A", "weight_label": "$w_{1}$", "ct": 0.1},
                {"group": "B", "weight_label": "$w_{2}$", "ct": 0.2},
            ]
        )

        tex = df_to_latex_table(
            df,
            caption="Caption text",
            label="tab:test",
            table_env=True,
            weights_in_columns=True,
            font_size=r"\tiny",
        )

        self.assertIn(r"\begin{table}", tex)
        self.assertIn(r"\tiny", tex)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\end{tabular}", tex)
        self.assertIn(r"\caption{Caption text}", tex)
        self.assertIn(r"\label{tab:test}", tex)
        self.assertLess(tex.index(r"\tiny"), tex.index(r"\begin{tabular}"))
        self.assertLess(tex.index(r"\end{tabular}"), tex.index(r"\caption{Caption text}"))
        self.assertLess(tex.index(r"\caption{Caption text}"), tex.index(r"\label{tab:test}"))

    def test_df_to_latex_table_uses_single_backslash_latex_commands(self):
        df = pd.DataFrame(
            [
                {"group": "A", "weight_label": "$w_{1}$", "ct": 0.1},
                {"group": "B", "weight_label": "$w_{2}$", "ct": 0.2},
            ]
        )

        tex = df_to_latex_table(
            df,
            caption="Caption text",
            label="tab:test",
            table_env=True,
            weights_in_columns=True,
            font_size=r"\tiny",
        )

        self.assertIn(r"\begin{table}", tex)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\toprule", tex)
        self.assertIn(r"\midrule", tex)
        self.assertIn(r"\bottomrule", tex)
        self.assertIn(r"\multicolumn", tex)
        self.assertNotIn(r"\\begin{table}", tex)
        self.assertNotIn(r"\\begin{tabular}", tex)
        self.assertNotIn(r"\\toprule", tex)
        self.assertNotIn(r"\\multicolumn", tex)


if __name__ == "__main__":
    unittest.main()
