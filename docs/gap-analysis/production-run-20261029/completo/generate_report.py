#!/usr/bin/env python3
"""Tables and figures for the report on the production run of 2026-10-29.

Three experiments over DATA_PRP_20C (150 instances x 6 weight vectors x alpha 0.99):

* ``pso_mip_polish``          PSO followed by the full MIP, polishing after 3000 s;
* ``matheuristic_mip_polish`` matheuristic followed by the full MIP, polishing after 3000 s;
* ``matheuristic``            matheuristic alone.

Run from the repository root::

    poetry run python docs/gap-analysis/production-run-20261029/completo/generate_report.py \
        --input-dir /Users/gabamaro/repos/lsvrp/out

It reads each experiment's ``*-union_results.xlsx`` and, for the matheuristic plus MIP
experiment, the stage records (``*.stages.parquet`` + ``*.metadata.json``), and writes
LaTeX fragments to ``generated/`` and figures to ``figures/``, next to this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    COLORS,
    EXPECTED_TASKS,
    EXPERIMENTS,
    LABELS,
    bootstrap_mean_ci,
    load_stages,
    load_union,
    num,
    pct,
    table,
    weight_label,
)


def write(name, content):
    path = HERE / "generated" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def quantile_row(series, label):
    q = series.quantile([0.1, 0.5, 0.9]).values
    return [label, str(int(series.notna().sum())), num(series.mean()), num(q[1]), num(q[0]), num(q[2]), num(series.max())]


# ---------------------------------------------------------------------------- report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="out")
    args = parser.parse_args()

    data, raw = {}, {}
    for key, name in EXPERIMENTS.items():
        data[key], raw[key] = load_union(args.input_dir, name)
    A, B, C = data["A"], data["B"], data["C"]
    stages = load_stages(args.input_dir)
    B = B.join(stages)
    all_tasks = A.index
    common = A.index.intersection(C.index)
    A_ok, B_ok = A.gap.notna(), B.gap.notna()
    clean = A.index[A_ok].intersection(B.index[B_ok]).intersection(C.index)
    missing = sorted(set(all_tasks) - set(C.index))

    macros = {
        "nTasks": EXPECTED_TASKS,
        "nCommon": len(common),
        "nClean": len(clean),
        "nMissingC": len(missing),
        "nCtasks": len(C),
        "nAfallback": int((~A_ok).sum()),
        "nBfallback": int((~B_ok).sum()),
        "pctAfallback": f"{100 * (~A_ok).mean():.0f}",
        "pctBfallback": f"{100 * (~B_ok).mean():.0f}",
        "nAgap": int(A_ok.sum()),
        "nBgap": int(B_ok.sum()),
        "pctBgainRejected": f"{100 * ((B.restricted_fo - B.mip_fo) / B.restricted_fo)[~B.mip_selected].mean():.1f}".replace(".", ","),
    }
    write("numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))

    # ---- execution summary
    rows = []
    for key in "ABC":
        t = data[key] if key != "B" else B
        final_from_mip = {"A": int(A_ok.sum()), "B": int(B_ok.sum()), "C": 0}[key]
        rows.append(
            [
                f"{key}. {LABELS[key]}",
                str(len(t)),
                str(EXPECTED_TASKS - len(t)),
                str(final_from_mip) if key != "C" else "--",
                num(t.solver_time.median() / 60, 1),
                num(t.solver_time.max() / 60, 1),
            ]
        )
    write(
        "tab_execucao.tex",
        table(
            ["Experimento", "Tarefas", "Faltantes", "Solu\u00e7\u00e3o final do MIP", "Tempo mediano (min)", "Tempo m\u00e1ximo (min)"],
            rows,
            spec="lrrrrr",
        ),
    )

    # ---- objective: distributions
    series = {
        "A": ("A. PSO + MIP (reportado)", A.FO),
        "B": ("B. Matheur\u00edstica + MIP (reportado)", B.FO),
        "Bm": ("B. Matheur\u00edstica + MIP (incumbente do MIP)", B.mip_fo),
        "C": ("C. Matheur\u00edstica", C.FO),
    }
    rows = [quantile_row(s, label) for _, (label, s) in series.items()]
    write("tab_fo_dist.tex", table(["Fluxo", "n", "M\u00e9dia", "Mediana", "P10", "P90", "M\u00e1x."], rows, spec="lrrrrrr"))

    rows = []
    for key in "ABC":
        s = data[key].loc[clean].FO if key != "B" else B.loc[clean].FO
        q = s.quantile([0.1, 0.5, 0.9]).values
        rows.append([f"{key}. {LABELS[key]}", num(s.mean()), num(q[1]), num(q[0]), num(q[2])])
    write("tab_fo_clean.tex", table(["Fluxo", "M\u00e9dia", "Mediana", "P10", "P90"], rows, spec="lrrrr"))

    # ---- paired comparisons
    def paired(x, y, index, label):
        d = ((x.loc[index] - y.loc[index]) / y.loc[index]).dropna()
        lo, hi = bootstrap_mean_ci(d.values)
        return [
            label,
            str(len(d)),
            pct((d < -1e-9).mean(), 0),
            pct((d.abs() <= 1e-9).mean(), 0),
            pct(d.median(), 2).replace("%", "\\%") if False else pct(d.median(), 2),
            pct(d.mean(), 2),
            f"[{pct(lo, 2)}; {pct(hi, 2)}]",
        ]

    cols = ["Compara\u00e7\u00e3o (1\u00ba contra 2\u00ba)", "n", "1\u00ba melhor", "Empate", "Mediana", "M\u00e9dia", "IC 95\\% da m\u00e9dia"]
    rows = [
        paired(A.FO, B.FO, common, "A contra B"),
        paired(A.FO, C.FO, common, "A contra C"),
        paired(B.FO, C.FO, common, "B contra C"),
        paired(B.mip_fo, C.FO, common, "B (MIP) contra C"),
        paired(A.FO, B.mip_fo, common, "A contra B (MIP)"),
    ]
    write("tab_pareada_todas.tex", table(cols, rows, spec="lrrrrrr"))
    rows = [
        paired(A.FO, B.FO, clean, "A contra B"),
        paired(A.FO, C.FO, clean, "A contra C"),
        paired(B.FO, C.FO, clean, "B contra C"),
    ]
    write("tab_pareada_limpo.tex", table(cols, rows, spec="lrrrrrr"))

    # ---- by weight vector and by class (reported FO)
    def by_group(group_col, label_fn):
        rows = []
        for value, idx in sorted(A.groupby(group_col).groups.items()):
            c_idx = [i for i in idx if i in C.index]
            rows.append(
                [
                    label_fn(value),
                    num(A.loc[idx].FO.mean()),
                    num(B.loc[idx].FO.mean()),
                    num(B.loc[idx].mip_fo.mean()),
                    num(C.loc[c_idx].FO.mean()),
                ]
            )
        return rows

    cols = ["", "A (reportado)", "B (reportado)", "B (MIP)", "C"]
    write("tab_fo_peso.tex", table(["Pesos $w_1,\\ldots,w_5$"] + cols[1:], by_group("weight", weight_label), spec="lrrrr"))
    write("tab_fo_classe.tex", table(["Classe"] + cols[1:], by_group("classe", lambda v: str(int(v))), spec="lrrrr"))

    # ---- components (totals over the horizon) on the clean subset
    rows = []
    for j, name in enumerate(["f1 (produ\u00e7\u00e3o)", "f2 (setup)", "f3 (estoque)", "f4 (transp. fixo)", "f5 (transp. vari\u00e1vel)"], start=1):
        col = f"f{j}"
        a, b, c = A.loc[clean][col].mean(), B.loc[clean][col].mean(), C.loc[clean][col].mean()
        rows.append([name, f"{a:,.0f}".replace(",", " "), pct(b / a - 1, 1), pct(c / a - 1, 1)])
    s_a, s_b, s_c = (x.loc[clean].total_setup.mean() for x in (A, B, C))
    rows.append(["n\u00ba de setups (soma de $y$)", num(s_a, 1), pct(s_b / s_a - 1, 1), pct(s_c / s_a - 1, 1)])
    write("tab_componentes.tex", table(["Componente (m\u00e9dia por tarefa)", "A", "B contra A", "C contra A"], rows, spec="lrrr"))

    # ---- effect of the MIP on the matheuristic (stage records)
    rows = []
    for label, mask in (("Solu\u00e7\u00e3o do MIP aceita", B.mip_selected), ("Solu\u00e7\u00e3o do MIP rejeitada", ~B.mip_selected), ("Todas", B.mip_selected | ~B.mip_selected)):
        gain = ((B.restricted_fo - B.mip_fo) / B.restricted_fo)[mask]
        rows.append(
            [label, str(int(mask.sum())), pct(gain.mean(), 1), pct(gain.median(), 1), pct(gain.quantile(0.9), 1), pct(gain.max(), 1)]
        )
    write("tab_ganho_mip.tex", table(["Grupo", "n", "M\u00e9dia", "Mediana", "P90", "M\u00e1x."], rows, spec="lrrrrr"))

    # ---- gap distributions
    def gap_row(series, label):
        s = series.dropna()
        return [
            label,
            str(len(s)),
            pct(s.mean()),
            pct(s.median()),
            pct(s.quantile(0.9)),
            pct(s.max()),
            pct((s <= 0.05).mean(), 0),
            pct((s <= 0.10).mean(), 0),
            pct((s <= 0.20).mean(), 0),
            str(int((s <= 1e-9).sum())),
        ]

    cols = ["Fonte do gap", "n", "M\u00e9dia", "Mediana", "P90", "M\u00e1x.", "$\\le 5\\%$", "$\\le 10\\%$", "$\\le 20\\%$", "Gap $0$"]
    rows = [
        gap_row(A.gap, "A (reportado, MIP aceito)"),
        gap_row(B.gap, "B (reportado, MIP aceito)"),
        gap_row(B.mip_gap, "B (MIP, todas as tarefas)"),
        gap_row(A.loc[clean].gap, "A (subconjunto limpo)"),
        gap_row(B.loc[clean].gap, "B (subconjunto limpo)"),
    ]
    write("tab_gap_dist.tex", table(cols, rows, spec="lrrrrrrrrr"))

    # ---- gap by weight and class
    def gap_by(group_col, label_fn):
        rows = []
        for value, idx in sorted(A.groupby(group_col).groups.items()):
            rows.append(
                [
                    label_fn(value),
                    pct(A.loc[idx].gap.mean()),
                    pct(B.loc[idx].gap.mean()),
                    pct(B.loc[idx].mip_gap.mean()),
                    pct(B.loc[idx].mip_gap.median()),
                    pct((~A_ok.loc[idx]).mean(), 0),
                    pct((~B_ok.loc[idx]).mean(), 0),
                ]
            )
        return rows

    cols = ["", "A (rep.)", "B (rep.)", "B (MIP) m\u00e9dia", "B (MIP) mediana", "A sem MIP", "B sem MIP"]
    write("tab_gap_peso.tex", table(["Pesos $w_1,\\ldots,w_5$"] + cols[1:], gap_by("weight", weight_label), spec="lrrrrrr"))
    write("tab_gap_classe.tex", table(["Classe"] + cols[1:], gap_by("classe", lambda v: str(int(v))), spec="lrrrrrr"))

    # ---- bound comparison (bound = FO * (1 - gap))
    bound_a = A.FO * (1 - A.gap)
    bound_b = B.FO * (1 - B.gap)
    rows = [
        ["A (subconjunto limpo)", num(bound_a.loc[clean].mean()), num(A.loc[clean].FO.mean()), pct(A.loc[clean].gap.mean())],
        ["B (subconjunto limpo)", num(bound_b.loc[clean].mean()), num(B.loc[clean].FO.mean()), pct(B.loc[clean].gap.mean())],
        ["B (MIP, todas as tarefas)", num(B.mip_bound.mean()), num(B.mip_fo.mean()), pct(B.mip_gap.mean())],
    ]
    write("tab_bound.tex", table(["Fonte", "Limite inferior m\u00e9dio", "Incumbente m\u00e9dia", "Gap m\u00e9dio"], rows, spec="lrrr"))
    ratio = (bound_a.loc[clean] / bound_b.loc[clean] - 1)
    write("numbers_bound.tex", f"\\newcommand{{\\boundDiffMedian}}{{{pct(ratio.median(), 2)}}}\n\\newcommand{{\\boundDiffMean}}{{{pct(ratio.mean(), 2)}}}\n")

    # ---- figures
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, s, color, style in (
        ("A (reportado, MIP aceito)", A.gap.dropna(), COLORS["A"], "-"),
        ("B (reportado, MIP aceito)", B.gap.dropna(), COLORS["B"], "-"),
        ("B (MIP, todas as tarefas)", B.mip_gap.dropna(), COLORS["B"], "--"),
    ):
        values = np.sort(s.values) * 100
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linestyle=style, label=label)
    ax.set_xlabel("Gap (%)")
    ax.set_ylabel("Fra\u00e7\u00e3o acumulada de tarefas")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "production-run-gap-cdf.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, x, color in (
        ("A contra C", (A.loc[common].FO - C.loc[common].FO) / C.loc[common].FO, COLORS["A"]),
        ("B contra C", (B.loc[common].FO - C.loc[common].FO) / C.loc[common].FO, COLORS["B"]),
        ("B (MIP) contra C", (B.loc[common].mip_fo - C.loc[common].FO) / C.loc[common].FO, "#7f3c8d"),
    ):
        values = np.sort(x.values) * 100
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, label=label)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Diferen\u00e7a relativa da FO em rela\u00e7\u00e3o \u00e0 matheur\u00edstica (%)")
    ax.set_ylabel("Fra\u00e7\u00e3o acumulada de tarefas")
    ax.set_xlim(-25, 25)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "production-run-fo-diff-cdf.pdf")
    plt.close(fig)

    write(
        "missing_tasks.tex",
        "\\begin{itemize}[leftmargin=*]\n"
        + "".join(
            f"  \\item \\texttt{{{key.split('|')[0].split('/')[-1].replace('_', chr(92) + '_')}}}, pesos "
            f"{weight_label(key.split('|')[1])}\n"
            for key in missing
        )
        + "\\end{itemize}\n",
    )
    # ---- didactic material: spread across tasks, tails of the paired differences, figure readings
    d_ac_clean = ((A.loc[clean].FO - C.loc[clean].FO) / C.loc[clean].FO)
    d_ac_all = ((A.loc[common].FO - C.loc[common].FO) / C.loc[common].FO)
    extra = {
        "sdFOA": num(A.FO.std()),
        "meanFOA": num(A.FO.mean()),
        "minFOA": num(A.FO.min()),
        "maxFOA": num(A.FO.max()),
        "sdDiffACclean": pct(d_ac_clean.std(), 1),
        "pNinetyDiffACall": pct(d_ac_all.quantile(0.9), 1),
        "shareWorseTenACall": pct((d_ac_all > 0.10).mean(), 0),
        "shareWorseTenACclean": pct((d_ac_clean > 0.10).mean(), 0),
        "meanDiffACall": pct(d_ac_all.mean(), 2),
        "medianDiffACall": pct(d_ac_all.median(), 2),
    }
    write("numbers_didatico.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in extra.items()))

    # reading of the cumulative curve of the FO difference at x = 0, +5% and +10%
    rows = []
    for label, x in (
        ("A contra C", (A.loc[common].FO - C.loc[common].FO) / C.loc[common].FO),
        ("B contra C", (B.loc[common].FO - C.loc[common].FO) / C.loc[common].FO),
        ("B (MIP) contra C", (B.loc[common].mip_fo - C.loc[common].FO) / C.loc[common].FO),
    ):
        rows.append([label, pct((x <= 1e-9).mean(), 0), pct((x <= 0.05).mean(), 0), pct((x <= 0.10).mean(), 0), pct((x > 0.10).mean(), 0)])
    write(
        "tab_fig_leitura.tex",
        table(["Curva", "$x\\le 0$ (melhor ou igual)", "$x\\le 5\\%$", "$x\\le 10\\%$", "Mais de 10\\% pior"], rows, spec="lrrrr"),
    )

    # worked example: the clean task whose A-versus-C difference is closest to the median
    target = d_ac_clean.sub(d_ac_clean.median()).abs().idxmin()
    example_rows = []
    for key, value in (("A", A.loc[target].FO), ("B", B.loc[target].FO), ("C", C.loc[target].FO)):
        diff = (value - C.loc[target].FO) / C.loc[target].FO
        example_rows.append([f"{key}. {LABELS[key]}", num(value), "--" if key == "C" else pct(diff, 2)])
    write("tab_exemplo.tex", table(["Fluxo", "FO da tarefa", "Diferen\u00e7a em rela\u00e7\u00e3o a C"], example_rows, spec="lrr"))
    instance = target.split("|")[0].split("/")[-1].replace(".dat", "").replace("_", "\\_")
    write("exemplo_tarefa.tex", f"\\texttt{{{instance}}} com pesos {weight_label(target.split('|')[1])}")

    print("written to", HERE / "generated")


if __name__ == "__main__":
    main()
