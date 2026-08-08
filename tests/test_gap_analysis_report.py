import unittest

import pandas as pd

from src.reports.gap_analysis import gap_table_to_latex_alpha_weight


class GapAnalysisReportTestCase(unittest.TestCase):
    def test_gap_table_to_latex_places_font_size_before_tabular_and_caption_after(self):
        tabela = pd.DataFrame(
            [[1.2, 2.3]],
            index=pd.MultiIndex.from_tuples([(5, "1")], names=["clientes", "classe"]),
            columns=pd.MultiIndex.from_tuples(
                [(0.01, "$w_{1}$"), (0.99, "$w_{2}$")], names=["alpha", "weight_label"]
            ),
        )

        tex = gap_table_to_latex_alpha_weight(
            tabela,
            caption="Gap caption",
            label="tab:gap",
            table_env=True,
            font_size=r"\Large",
        )

        self.assertIn(r"\begin{table}", tex)
        self.assertIn(r"\Large", tex)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\end{tabular}", tex)
        self.assertIn(r"\caption{Gap caption}", tex)
        self.assertIn(r"\label{tab:gap}", tex)
        self.assertLess(tex.index(r"\Large"), tex.index(r"\begin{tabular}"))
        self.assertLess(tex.index(r"\end{tabular}"), tex.index(r"\caption{Gap caption}"))
        self.assertLess(tex.index(r"\caption{Gap caption}"), tex.index(r"\label{tab:gap}"))

    def test_gap_table_to_latex_renders_header_rules_and_multirow_layout(self):
        tabela = pd.DataFrame(
            [[1.2, 2.3], [3.4, 4.5], [5.6, 6.7], [7.8, 8.9]],
            index=pd.MultiIndex.from_tuples(
                [(5, "1"), (5, "all"), (7, "1"), (7, "all")],
                names=["clientes", "classe"],
            ),
            columns=pd.MultiIndex.from_tuples(
                [(0.01, "$w_{1}$"), (0.99, "$w_{2}$")],
                names=["alpha", "weight_label"],
            ),
        )

        tex = gap_table_to_latex_alpha_weight(
            tabela,
            clientes_order=[5, 7],
            classes_order=["1", "all"],
            alpha_order=[0.01, 0.99],
            weight_order=["$w_{1}$", "$w_{2}$"],
            table_env=False,
        )

        self.assertIn(r"\multicolumn{2}{c}{\textbf{$\alpha=0.01$}}", tex)
        self.assertIn(r"\multicolumn{2}{c}{\textbf{$\alpha=0.99$}}", tex)
        self.assertIn(r"\cmidrule(lr){3-4}", tex)
        self.assertIn(r"\cmidrule(lr){5-6}", tex)
        self.assertIn(r"\multirow{2}{*}{\textbf{5C}}", tex)
        self.assertIn(r"\multirow{2}{*}{\textbf{7C}}", tex)

        lines = tex.splitlines()
        first_client_row = next(i for i, line in enumerate(lines) if r"\multirow{2}{*}{\textbf{5C}}" in line)
        second_block_rule = next(
            i for i, line in enumerate(lines) if line == r"\midrule" and i > first_client_row
        )
        self.assertLess(first_client_row, second_block_rule)


if __name__ == "__main__":
    unittest.main()
