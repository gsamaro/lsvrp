#!/usr/bin/env python3
"""Paired-only tables and figures for the production run of 2026-10-29.

Only the tasks that can be compared across the three methods are analysed: those in which the
MIP solution was accepted by the validation in both PSO + MIP (A) and matheuristic + MIP (B) and
that also exist in the matheuristic-only experiment (C). Everything else is discarded and not shown.

Run from the repository root::

    poetry run python docs/gap-analysis/production-run-20261029/pareado/generate_report.py \
        --input-dir /Users/gabamaro/repos/lsvrp/out

Writes LaTeX fragments to ``generated/`` and figures to ``figures/``, next to this script.
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

OUT_DIR = HERE / "generated"
PAIRED_COLORS = {"AC": COLORS["A"], "BC": COLORS["B"], "AB": "#7f3c8d"}


def write(name, content):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(content, encoding="utf-8")


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def rel_diff(x, y):
    return (x - y) / y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="out")
    args = parser.parse_args()

    data = {key: load_union(args.input_dir, name)[0] for key, name in EXPERIMENTS.items()}
    A, B, C = data["A"], data["B"].join(load_stages(args.input_dir)), data["C"]
    paired = A.index[A.gap.notna()].intersection(B.index[B.gap.notna()]).intersection(C.index)
    A, B, C = A.loc[paired], B.loc[paired], C.loc[paired]
    n = len(paired)
    executed = {"A": len(data["A"]), "B": len(data["B"]), "C": len(data["C"])}

    # ------------------------------------------------------------------ numbers
    d_ac = rel_diff(A.FO, C.FO)
    sd_fo = A.FO.std()
    se_paired = d_ac.std() / np.sqrt(n)
    se_unpaired = (sd_fo / A.FO.mean()) * np.sqrt(2 / n)
    numbers = (
        macro("nTasks", 900)
        + macro("nPaired", n)
        + macro("nExcluded", 900 - n)
        + macro("sdFOA", num(sd_fo))
        + macro("meanFOA", num(A.FO.mean()))
        + macro("cvFOA", pct(sd_fo / A.FO.mean(), 0))
        + macro("minFOA", num(A.FO.min()))
        + macro("maxFOA", num(A.FO.max()))
        + macro("sdDiffAC", pct(d_ac.std(), 1))
        + macro("sePaired", f"{100 * se_paired:.1f}".replace(".", ","))
        + macro("seUnpaired", f"{100 * se_unpaired:.1f}".replace(".", ","))
    )
    write("numbers.tex", numbers)

    # ---------------------------------------------------------------- execution
    rows = [
        [f"{key}. {LABELS[key]}", str(executed[key]), str(n)] for key in "ABC"
    ]
    write("tab_execucao.tex", table(["Método", "Tarefas com resultado", "Tarefas analisadas (pareadas)"], rows, spec="lrr"))

    # ---------------------------------------------------------------- FO
    def quantile_row(series, label):
        q = series.quantile([0.1, 0.5, 0.9]).values
        return [label, str(len(series)), num(series.mean()), num(q[1]), num(q[0]), num(q[2]), num(series.max())]

    rows = [quantile_row(t.FO, f"{key}. {LABELS[key]}") for key, t in (("A", A), ("B", B), ("C", C))]
    write("tab_fo_dist.tex", table(["Método", "n", "Média", "Mediana", "P10", "P90", "Máx."], rows, spec="lrrrrrr"))

    # ---------------------------------------------------------------- paired FO
    def paired_row(x, y, label):
        d = rel_diff(x, y)
        lo, hi = bootstrap_mean_ci(d.values)
        return [
            label,
            str(len(d)),
            pct((d < -1e-9).mean(), 0),
            pct((d.abs() <= 1e-9).mean(), 0),
            pct(d.median(), 2),
            pct(d.mean(), 2),
            f"[{pct(lo, 2)}; {pct(hi, 2)}]",
        ]

    cols = ["Comparação (1º contra 2º)", "n", "1º melhor", "Empate", "Mediana", "Média", "IC 95\\% da média"]
    comparisons = (("A contra B", A.FO, B.FO, "AB"), ("A contra C", A.FO, C.FO, "AC"), ("B contra C", B.FO, C.FO, "BC"))
    write("tab_pareada.tex", table(cols, [paired_row(x, y, label) for label, x, y, _ in comparisons], spec="lrrrrrr"))

    # worked example: the task whose A-versus-C difference is closest to the median
    target = d_ac.sub(d_ac.median()).abs().idxmin()
    rows = []
    for key, t in (("A", A), ("B", B), ("C", C)):
        diff = "--" if key == "C" else pct(rel_diff(t.loc[target].FO, C.loc[target].FO), 2)
        rows.append([f"{key}. {LABELS[key]}", num(t.loc[target].FO), diff])
    write("tab_exemplo.tex", table(["Método", "FO da tarefa", "Diferença em relação a C"], rows, spec="lrr"))
    instance = target.split("|")[0].split("/")[-1].replace(".dat", "").replace("_", "\\_")
    write("exemplo_tarefa.tex", f"\\texttt{{{instance}}} com pesos {weight_label(target.split('|')[1])}")

    # figure reading
    rows = []
    for label, x, y, _ in comparisons:
        d = rel_diff(x, y)
        rows.append([label] + [pct((d <= t + 1e-9).mean(), 0) for t in (-0.05, 0.0, 0.05, 0.10)])
    write(
        "tab_fig_leitura.tex",
        table(["Curva", "$x\\le -5\\%$", "$x\\le 0$ (melhor ou igual)", "$x\\le +5\\%$", "$x\\le +10\\%$"], rows, spec="lrrrr"),
    )

    # ---------------------------------------------------------------- by weight / class
    def by_group(group_col, label_fn):
        rows = []
        for value, idx in sorted(A.groupby(group_col).groups.items()):
            rows.append([label_fn(value), str(len(idx)), num(A.loc[idx].FO.mean()), num(B.loc[idx].FO.mean()), num(C.loc[idx].FO.mean())])
        return rows

    write("tab_fo_peso.tex", table(["Pesos $w_1,\\ldots,w_5$", "n", "A", "B", "C"], by_group("weight", weight_label), spec="lrrrr"))
    write("tab_fo_classe.tex", table(["Classe", "n", "A", "B", "C"], by_group("classe", lambda v: str(int(v))), spec="lrrrr"))

    # ---------------------------------------------------------------- components
    rows = []
    names = ["f1 (produção)", "f2 (setup)", "f3 (estoque)", "f4 (transp. fixo)", "f5 (transp. variável)"]
    for j, name in enumerate(names, start=1):
        col = f"f{j}"
        a, b, c = A[col].mean(), B[col].mean(), C[col].mean()
        rows.append([name, f"{a:,.0f}".replace(",", " "), pct(b / a - 1, 1), pct(c / a - 1, 1)])
    s_a, s_b, s_c = A.total_setup.mean(), B.total_setup.mean(), C.total_setup.mean()
    rows.append(["nº de setups (soma de $y$)", num(s_a, 1), pct(s_b / s_a - 1, 1), pct(s_c / s_a - 1, 1)])
    write("tab_componentes.tex", table(["Componente (média por tarefa)", "A", "B contra A", "C contra A"], rows, spec="lrrr"))

    # ---------------------------------------------------------------- MIP gain over the matheuristic (B)
    gain = (B.restricted_fo - B.mip_fo) / B.restricted_fo
    q = gain.quantile([0.1, 0.5, 0.9]).values
    lo, hi = bootstrap_mean_ci(gain.values)
    write(
        "tab_ganho_mip.tex",
        table(
            ["n", "Média", "IC 95\\% da média", "P10", "Mediana", "P90", "Máx."],
            [[str(len(gain)), pct(gain.mean(), 1), f"[{pct(lo, 1)}; {pct(hi, 1)}]", pct(q[0], 1), pct(q[1], 1), pct(q[2], 1), pct(gain.max(), 1)]],
            spec="rrrrrrr",
        ),
    )
    write("numbers_ganho.tex", macro("gainMedian", pct(q[1], 1)) + macro("gainMax", pct(gain.max(), 0)))

    # ---------------------------------------------------------------- gap
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

    cols = ["Método", "n", "Média", "Mediana", "P90", "Máx.", "$\\le 5\\%$", "$\\le 10\\%$", "$\\le 20\\%$", "Gap $0$"]
    write("tab_gap_dist.tex", table(cols, [gap_row(A.gap, "A. PSO + MIP"), gap_row(B.gap, "B. Matheurística + MIP")], spec="lrrrrrrrrr"))

    # paired gap difference in percentage points (A - B); negative means A has the smaller gap
    dg = (A.gap - B.gap) * 100
    lo, hi = bootstrap_mean_ci(dg.values)
    write(
        "tab_gap_pareado.tex",
        table(
            ["Comparação", "n", "A menor", "Empate", "Mediana (p.p.)", "Média (p.p.)", "IC 95\\% da média (p.p.)"],
            [[
                "gap de A contra gap de B",
                str(len(dg)),
                pct((dg < -1e-9).mean(), 0),
                pct((dg.abs() <= 1e-9).mean(), 0),
                f"{dg.median():.2f}".replace(".", ","),
                f"{dg.mean():.2f}".replace(".", ","),
                f"[{lo:.2f}; {hi:.2f}]".replace(".", ","),
            ]],
            spec="lrrrrrr",
        ),
    )

    bound_a, bound_b = A.FO * (1 - A.gap), B.FO * (1 - B.gap)
    ratio = bound_a / bound_b - 1
    rows = [
        ["A. PSO + MIP", num(bound_a.mean()), num(A.FO.mean()), pct(A.gap.mean())],
        ["B. Matheurística + MIP", num(bound_b.mean()), num(B.FO.mean()), pct(B.gap.mean())],
    ]
    write("tab_bound.tex", table(["Método", "Limite inferior médio", "Incumbente média", "Gap médio"], rows, spec="lrrr"))
    write("numbers_bound.tex", macro("boundDiffMedian", pct(ratio.median(), 2)) + macro("boundDiffMean", pct(ratio.mean(), 2)))

    def gap_by(group_col, label_fn):
        rows = []
        for value, idx in sorted(A.groupby(group_col).groups.items()):
            rows.append(
                [
                    label_fn(value),
                    str(len(idx)),
                    pct(A.loc[idx].gap.mean()),
                    pct(B.loc[idx].gap.mean()),
                    pct(A.loc[idx].gap.median()),
                    pct(B.loc[idx].gap.median()),
                ]
            )
        return rows

    cols = ["", "n", "A (média)", "B (média)", "A (mediana)", "B (mediana)"]
    write("tab_gap_peso.tex", table(["Pesos $w_1,\\ldots,w_5$"] + cols[1:], gap_by("weight", weight_label), spec="lrrrrr"))
    write("tab_gap_classe.tex", table(["Classe"] + cols[1:], gap_by("classe", lambda v: str(int(v))), spec="lrrrrr"))

    # ---------------------------------------------------------------- figures
    fig_dir = HERE / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, s, color in (("A. PSO + MIP", A.gap, COLORS["A"]), ("B. Matheurística + MIP", B.gap, COLORS["B"])):
        values = np.sort(s.values) * 100
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, label=label)
    ax.set_xlabel("Gap (%)")
    ax.set_ylabel("Fração acumulada de tarefas")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "production-run-pareado-gap-cdf.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, x, y, key in comparisons:
        values = np.sort(rel_diff(x, y).values) * 100
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=PAIRED_COLORS[key], label=label)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Diferença relativa da FO entre os dois métodos (%)")
    ax.set_ylabel("Fração acumulada de tarefas")
    ax.set_xlim(-30, 30)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "production-run-pareado-fo-diff-cdf.pdf")
    plt.close(fig)

    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
