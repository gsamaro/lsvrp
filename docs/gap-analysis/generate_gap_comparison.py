#!/usr/bin/env python3
"""Recreate the DATA_PRP_20C gap tables and histogram/CDF figures.

Run from the repository root, for example:

    poetry run python docs/gap-analysis/generate_gap_comparison.py \
        --input-dir /Users/gabamaro/repos/lsvrp/out

By default, the script reads the union_results workbooks from ``out/``
and writes CSV data, LaTeX table fragments, and PNG/PDF figures to
``docs/gap-analysis/generated/``. Input locations and output directory can be overridden
with the command-line options shown by ``--help``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from constants import WEIGHTS_OPTIMIZE  # noqa: E402


DEFAULT_INPUT_FILENAMES = {
    "baseline": "2026-09-21-40da15-92914b-union_results.xlsx",
    "tight_bounds": "2026-09-19-898e7a-e7a9b4-union_results.xlsx",
    "all_features": "2026-09-16-898e7a-200533-union_results.xlsx",
    "october_run": "2026-10-05-0a58f8-22bd43-union_results.xlsx",
}
SCENARIOS = [
    ("baseline", "Baseline; modelo inicial do artigo, sem alterações", "#7546b8"),
    ("tight_bounds", "Bounds apertados; sem outras features", "#2457a6"),
    ("all_features", "Todas as features + bounds apertados", "#b86512"),
    ("october_run", "Monoobjetivo; execução de 05/10/2026", "#16806a"),
]
REQUIRED_COLUMNS = {
    "file",
    "instancia",
    "clientes",
    "classe",
    "seeds",
    "weight",
    "weight_hash",
    "alpha",
    "gap",
}
PERCENTILES = [0.25, 0.50, 0.75, 0.90]
PERCENTILE_COLUMNS = ["Q25", "Mediana", "Q75", "Q90"]
HISTOGRAM_EDGES = np.arange(0, 102, 2, dtype=float)  # 0..100, width 2
CDF_X = np.arange(0, 100.5, 0.5, dtype=float)
TEX_ROW_END = " " + chr(92) * 2


def _canonical_weight(value: Any) -> tuple[float, ...]:
    """Convert a workbook weight value to a stable numeric tuple."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Não foi possível interpretar o peso {value!r}.") from exc

    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Formato de peso inesperado: {value!r}.")
    try:
        result = tuple(round(float(item), 8) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valores inválidos no peso {value!r}.") from exc
    if len(result) != 5:
        raise ValueError(f"O peso deve ter cinco componentes: {value!r}.")
    return result


WEIGHT_ORDER = [_canonical_weight(weight) for weight in WEIGHTS_OPTIMIZE]
WEIGHT_LABELS = {weight: f"w_{index}" for index, weight in enumerate(WEIGHT_ORDER, 1)}
MULTIOBJECTIVE_WEIGHTS = set(WEIGHT_ORDER)
UNIT_WEIGHTS = {
    tuple(1.0 if index == unit else 0.0 for index in range(5))
    for unit in range(5)
}


def _read_scenario(path: Path, scenario_id: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Planilha não encontrada: {path}")

    frame = pd.read_excel(path, engine="openpyxl")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name}: faltam colunas necessárias: {', '.join(sorted(missing))}."
        )

    frame = frame.copy()
    frame["clientes"] = pd.to_numeric(frame["clientes"], errors="coerce")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame["gap"] = pd.to_numeric(frame["gap"], errors="coerce")
    frame = frame.dropna(
        subset=["file", "instancia", "clientes", "classe", "seeds", "alpha", "gap"]
    )
    frame = frame.loc[frame["clientes"].eq(20)].copy()
    if frame.empty:
        raise ValueError(f"{path.name}: não há linhas válidas de instâncias com 20 clientes.")

    frame["weight_tuple"] = frame["weight"].map(_canonical_weight)
    unexpected = set(frame["weight_tuple"]) - MULTIOBJECTIVE_WEIGHTS - UNIT_WEIGHTS
    if unexpected:
        formatted = ", ".join(str(weight) for weight in sorted(unexpected))
        raise ValueError(f"{path.name}: pesos fora do protocolo da seção 16: {formatted}.")

    # The October workbook represents the monoobjective run, but its unit
    # weights have no mip_relative_gap. Per the report definition, use the six
    # non-unit, multiobjective weight vectors from that workbook for GAP.
    allowed_weights = MULTIOBJECTIVE_WEIGHTS
    label_map = WEIGHT_LABELS
    frame = frame.loc[frame["weight_tuple"].isin(allowed_weights)].copy()
    if frame.empty:
        raise ValueError(f"{path.name}: não restaram pesos multiobjetivo para calcular o GAP.")

    frame["gap"] = frame["gap"].clip(lower=0)
    frame["gap_pct"] = frame["gap"] * 100.0
    frame["weight_label"] = frame["weight_tuple"].map(label_map)
    frame["scenario_id"] = scenario_id

    # ProcessResults writes one row per period. GAP is repeated across periods;
    # retain one observation for each instance, class, weight, alpha, and seed.
    keys = ["instancia", "classe", "seeds", "weight_hash", "alpha"]
    gap_counts = frame.groupby(keys, dropna=False)["gap_pct"].nunique(dropna=False)
    inconsistent = gap_counts[gap_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{path.name}: GAP varia entre as linhas de período para uma mesma execução."
        )
    frame = frame.drop_duplicates(subset=keys, keep="first").copy()

    return frame[
        [
            "scenario_id",
            "file",
            "instancia",
            "clientes",
            "classe",
            "seeds",
            "weight",
            "weight_hash",
            "weight_tuple",
            "weight_label",
            "alpha",
            "gap_pct",
        ]
    ]


def _summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.quantile(0.50)),
        "q75": float(values.quantile(0.75)),
        "q90": float(values.quantile(0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _make_summaries(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    weight_rows = []
    for scenario_id, label, _ in SCENARIOS:
        scenario = data.loc[data["scenario_id"].eq(scenario_id)]
        aggregate_rows.append(
            {"scenario_id": scenario_id, "configuration": label, **_summary(scenario["gap_pct"])}
        )
        by_weight = scenario.groupby("weight_label", sort=False)["gap_pct"]
        for weight_label in WEIGHT_LABELS.values():
            if weight_label not in by_weight.groups:
                continue
            weight_rows.append(
                {
                    "scenario_id": scenario_id,
                    "configuration": label,
                    "weight": weight_label,
                    **_summary(by_weight.get_group(weight_label)),
                }
            )

    return pd.DataFrame(aggregate_rows), pd.DataFrame(weight_rows)


def _make_paired_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["instancia", "classe", "seeds", "weight_hash", "alpha"]
    indexed: dict[str, pd.Series] = {}
    comparable_scenarios = SCENARIOS
    for scenario_id, label, _ in comparable_scenarios:
        scenario = data.loc[data["scenario_id"].eq(scenario_id)].set_index(keys)["gap_pct"]
        if scenario.index.has_duplicates:
            raise ValueError(f"{label}: combinações repetidas após remover as linhas de período.")
        indexed[scenario_id] = scenario.sort_index()

    base_index = indexed["baseline"].index
    for scenario_id, _, _ in comparable_scenarios[1:]:
        if not base_index.equals(indexed[scenario_id].index):
            missing_from = len(base_index.difference(indexed[scenario_id].index))
            extra_in = len(indexed[scenario_id].index.difference(base_index))
            raise ValueError(
                "Os cenários não têm as mesmas combinações para a comparação pareada "
                f"(baseline ausente no cenário {scenario_id}: {missing_from}; "
                f"combinações extras: {extra_in})."
            )

    comparisons = []
    for left_index, (left_id, left_label, _) in enumerate(comparable_scenarios):
        for right_index, (right_id, right_label, _) in enumerate(comparable_scenarios):
            if left_index >= right_index:
                continue
            differences = indexed[right_id] - indexed[left_id]
            comparisons.append(
                {
                    "comparison": f"{left_label} → {right_label}",
                    "left_scenario": left_id,
                    "right_scenario": right_id,
                    "n_paired": int(differences.size),
                    "gap_lower_count": int((differences < 0).sum()),
                    "gap_lower_pct": float((differences < 0).mean() * 100.0),
                    "mean_difference_pp": float(differences.mean()),
                }
            )
    return pd.DataFrame(comparisons)


def _format_number(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _format_summary_cell(value: float, percent: bool = True) -> str:
    formatted = _format_number(value)
    return f"{formatted}\\%" if percent else formatted


def _write_latex_tables(
    aggregate: pd.DataFrame,
    by_weight: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    columns = ["n", "mean", "q25", "median", "q75", "q90", "min", "max"]
    headers = ["$n$", "Média", "$Q_{25}$", "Mediana", "$Q_{75}$", "$Q_{90}$", "Mín.", "Máx."]

    aggregate_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        "Configuração & " + " & ".join(headers) + TEX_ROW_END,
        r"\midrule",
    ]
    for row in aggregate.itertuples(index=False):
        values = [str(row.n)] + [
            _format_summary_cell(getattr(row, column)) for column in columns[1:]
        ]
        aggregate_lines.append(row.configuration + " & " + " & ".join(values) + TEX_ROW_END)
    aggregate_lines.extend([r"\bottomrule", r"\end{tabular}", "}%"])
    (output_dir / "aggregate_table.tex").write_text(
        "\n".join(aggregate_lines) + "\n", encoding="utf-8"
    )

    weight_lines: list[str] = []
    for scenario_id, label, _ in SCENARIOS:
        weight_lines.extend(
            [
                f"\\paragraph{{{label}.}}",
                r"\begin{center}\small",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{lrrrrrrrr}",
                r"\toprule",
                "Peso & " + " & ".join(headers) + TEX_ROW_END,
                r"\midrule",
            ]
        )
        selected = by_weight.loc[by_weight["scenario_id"].eq(scenario_id)]
        for row in selected.itertuples(index=False):
            values = [str(row.n)] + [
                _format_summary_cell(getattr(row, column)) for column in columns[1:]
            ]
            weight_label = "$" + row.weight.replace("_", "_{") + "}$"
            weight_lines.append(weight_label + " & " + " & ".join(values) + TEX_ROW_END)
        weight_lines.extend(
            [r"\bottomrule", r"\end{tabular}", "}%", r"\end{center}", ""]
        )
    (output_dir / "weight_tables.tex").write_text(
        "\n".join(weight_lines), encoding="utf-8"
    )

    paired_lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Comparação & Pares & GAP menor & Pares com GAP menor (\\%) & Diferença média (p.p.) "
        + TEX_ROW_END,
        r"\midrule",
    ]
    for row in paired.itertuples(index=False):
        left = next(label for scenario_id, label, _ in SCENARIOS if scenario_id == row.left_scenario)
        right = next(label for scenario_id, label, _ in SCENARIOS if scenario_id == row.right_scenario)
        pair_label = f"{left} $\\rightarrow$ {right}"
        paired_lines.append(
            f"{pair_label} & {row.n_paired} & {row.gap_lower_count} & "
            f"{_format_summary_cell(row.gap_lower_pct)} & "
            f"{_format_number(row.mean_difference_pp)} "
            + TEX_ROW_END
        )
    paired_lines.extend([r"\bottomrule", r"\end{tabular}", "}%"])
    (output_dir / "paired_comparisons.tex").write_text(
        "\n".join(paired_lines) + "\n", encoding="utf-8"
    )


def _plot_scenario(
    values: pd.Series, scenario_id: str, label: str, color: str, figure_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numbers = values.to_numpy(dtype=float)
    counts, _ = np.histogram(numbers, bins=HISTOGRAM_EDGES)
    frequencies = counts / numbers.size * 100.0
    sorted_values = np.sort(numbers)
    cdf_counts = np.searchsorted(sorted_values, CDF_X, side="right")
    cdf_percent = cdf_counts / numbers.size * 100.0

    histogram_data = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "bin_lower": HISTOGRAM_EDGES[:-1],
            "bin_upper": HISTOGRAM_EDGES[1:],
            "bin_center": (HISTOGRAM_EDGES[:-1] + HISTOGRAM_EDGES[1:]) / 2,
            "count": counts,
            "frequency_pct": frequencies,
        }
    )
    cdf_data = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "gap_pct": CDF_X,
            "cumulative_count": cdf_counts,
            "cumulative_pct": cdf_percent,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(
        histogram_data["bin_center"],
        histogram_data["frequency_pct"],
        width=1.7,
        color=color,
        edgecolor=color,
        alpha=0.5,
        label="Frequência por faixa",
        zorder=2,
    )
    ax.step(
        CDF_X,
        cdf_percent,
        where="post",
        color="#172033",
        linewidth=1.8,
        label="CDF",
        zorder=3,
    )
    ax.set(
        title=label,
        xlabel="GAP (%)",
        ylabel="Execuções (%)",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, which="major", color="#dce3ec", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_dir / f"gap_{scenario_id}.png", dpi=180)
    plt.close(fig)
    return histogram_data, cdf_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera as tabelas e gráficos de GAP da comparação DATA_PRP_20C."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPOSITORY_ROOT / "out",
        help="Pasta com os quatro arquivos union_results.xlsx.",
    )
    parser.add_argument("--baseline", type=Path, help="Sobrescreve o arquivo Baseline.")
    parser.add_argument("--tight-bounds", type=Path, help="Sobrescreve o arquivo de bounds apertados.")
    parser.add_argument("--all-features", type=Path, help="Sobrescreve o arquivo com todas as features.")
    parser.add_argument("--october-run", type=Path, help="Sobrescreve a execução de 05/10/2026.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/gap-analysis/generated",
        help="Diretório para CSV e fragmentos LaTeX.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs/gap-analysis/figures",
        help="Diretório para os gráficos PNG usados pelo documento.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = {
        scenario_id: getattr(args, option_name)
        or args.input_dir / filename
        for scenario_id, filename, option_name in [
            ("baseline", DEFAULT_INPUT_FILENAMES["baseline"], "baseline"),
            ("tight_bounds", DEFAULT_INPUT_FILENAMES["tight_bounds"], "tight_bounds"),
            ("all_features", DEFAULT_INPUT_FILENAMES["all_features"], "all_features"),
            ("october_run", DEFAULT_INPUT_FILENAMES["october_run"], "october_run"),
        ]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        _read_scenario(path, scenario_id)
        for scenario_id, path in inputs.items()
    ]
    data = pd.concat(frames, ignore_index=True)
    aggregate, by_weight = _make_summaries(data)
    paired = _make_paired_comparisons(data)

    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    by_weight.to_csv(args.output_dir / "weight_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    _write_latex_tables(aggregate, by_weight, paired, args.output_dir)

    histogram_frames = []
    cdf_frames = []
    for scenario_id, label, color in SCENARIOS:
        values = data.loc[data["scenario_id"].eq(scenario_id), "gap_pct"]
        histogram, cdf = _plot_scenario(
            values, scenario_id, label, color, args.figure_dir
        )
        histogram_frames.append(histogram)
        cdf_frames.append(cdf)
    pd.concat(histogram_frames, ignore_index=True).to_csv(
        args.output_dir / "histogram_data.csv", index=False
    )
    pd.concat(cdf_frames, ignore_index=True).to_csv(
        args.output_dir / "cdf_data.csv", index=False
    )

    manifest = {
        "sources": {key: str(path) for key, path in inputs.items()},
        "output_dir": str(args.output_dir),
        "figure_dir": str(args.figure_dir),
        "filters": {
            "clients": 20,
            "reported_gap_weights": [list(weight) for weight in WEIGHT_ORDER],
            "excluded_single_objective_weights": [list(weight) for weight in sorted(UNIT_WEIGHTS)],
            "gap_processing": "clip below zero, then convert fraction to percent",
            "period_rows": "deduplicated by instance, class, seed, weight hash, and alpha",
            "october_run": "monoobjective run; report GAP only for non-unit weight vectors",
            "paired_comparisons": "all four scenarios, matched by instance, class, seed, weight hash, and alpha",
        },
        "observations_per_scenario": {
            row.scenario_id: int(row.n) for row in aggregate.itertuples(index=False)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Tabelas e gráficos gravados em: {args.output_dir}")
    print(aggregate[["configuration", "n", "mean", "q25", "median", "q75", "q90", "min", "max"]].to_string(index=False))


if __name__ == "__main__":
    main()
