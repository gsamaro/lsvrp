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


if __name__ == "__main__":
    unittest.main()
