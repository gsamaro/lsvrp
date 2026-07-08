import unittest

import pandas as pd

from src.reports.performance_weights import (
    _wins_table_to_latex,
    _winners_by_tau_to_latex,
)


class PerformanceWeightsReportTestCase(unittest.TestCase):
    def test_wins_table_to_latex_places_font_size_before_tabular_and_caption_after(self):
        win_counts = pd.DataFrame(
            [
                {"clientes": 5, "classe": 1, "weight_label": "$w_{1}$", "win_count": 3},
                {"clientes": 5, "classe": "all", "weight_label": "$w_{1}$", "win_count": 5},
            ]
        )

        tex = _wins_table_to_latex(
            win_counts,
            clientes_order=[5],
            classes_order=[1],
            weight_order=["$w_{1}$"],
            caption="Wins caption",
            label="tab:wins",
            table_env=True,
            font_size=r"\footnotesize",
        )

        self.assertIn(r"\begin{table}", tex)
        self.assertIn(r"\footnotesize", tex)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\end{tabular}", tex)
        self.assertIn(r"\caption{Wins caption}", tex)
        self.assertIn(r"\label{tab:wins}", tex)
        self.assertLess(tex.index(r"\footnotesize"), tex.index(r"\begin{tabular}"))
        self.assertLess(tex.index(r"\end{tabular}"), tex.index(r"\caption{Wins caption}"))
        self.assertLess(tex.index(r"\caption{Wins caption}"), tex.index(r"\label{tab:wins}"))

    def test_winners_by_tau_to_latex_places_font_size_before_tabular_and_caption_after(self):
        winners = pd.DataFrame(
            [
                {
                    "alpha": 0.01,
                    "clientes": 5,
                    "classe": 1,
                    "tau": 1.0,
                    "weight_label": "$w_{1}$",
                }
            ]
        )

        tex = _winners_by_tau_to_latex(
            winners,
            taus_order=[1.0],
            alpha_order=[0.01],
            clientes_order=[5],
            classes_order=[1],
            caption="Tau caption",
            label="tab:tau",
            table_env=True,
            font_size=r"\small",
        )

        self.assertIn(r"\begin{table}", tex)
        self.assertIn(r"\small", tex)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\end{tabular}", tex)
        self.assertIn(r"\caption{Tau caption}", tex)
        self.assertIn(r"\label{tab:tau}", tex)
        self.assertLess(tex.index(r"\small"), tex.index(r"\begin{tabular}"))
        self.assertLess(tex.index(r"\end{tabular}"), tex.index(r"\caption{Tau caption}"))
        self.assertLess(tex.index(r"\caption{Tau caption}"), tex.index(r"\label{tab:tau}"))


if __name__ == "__main__":
    unittest.main()
