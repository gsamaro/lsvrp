#!/usr/bin/env python3
"""Consolida shards de telemetria e gera performance profiles Plotly."""
import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def read_shards(root, suffix):
    paths = sorted(Path(root).rglob(f"*{suffix}.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False) if paths else pd.DataFrame()


def pairs(summary):
    keys = ["experiment_id", "instance_file", "weight", "alpha"]
    usable = summary[summary["solver_variant"].isin(["solver", "pso_mip_start"])].copy()
    grouped = usable.groupby(keys, dropna=False)["solver_variant"].agg(lambda values: set(values))
    valid_index = grouped[grouped.apply(lambda values: {"solver", "pso_mip_start"}.issubset(values))].index
    valid = usable.set_index(keys).loc[valid_index].reset_index() if len(valid_index) else usable.iloc[0:0]
    missing = grouped[grouped.apply(lambda values: not {"solver", "pso_mip_start"}.issubset(values))]
    return valid, missing.reset_index(name="variants")


def profile_figure(valid, metric, title):
    rows = []
    keys = ["experiment_id", "instance_file", "weight", "alpha"]
    for _, group in valid.groupby(keys, dropna=False):
        eligible = group[group[metric].notna()]
        if len(eligible) < 2:
            continue
        best = eligible[metric].min()
        for _, row in eligible.iterrows():
            rows.append((row["solver_variant"], float(row[metric]) / best))
    fig = go.Figure()
    for strategy in ("solver", "pso_mip_start"):
        ratios = sorted(value for name, value in rows if name == strategy)
        if ratios:
            fig.add_trace(go.Scatter(x=ratios, y=[(idx + 1) / len(ratios) for idx in range(len(ratios))], mode="lines+markers", name=strategy))
    fig.update_layout(title=title, template="plotly_white", xaxis_title="Razão de desempenho (menor é melhor)", yaxis_title="Fração acumulada de instâncias", yaxis=dict(range=[0, 1]))
    return fig


def success_figure(valid):
    metrics = {"Primeira factível": "first_feasible_seconds", "Gap-alvo": "gap_target_seconds", "Execução concluída": "completed_total_seconds"}
    fig = go.Figure()
    for strategy in ("solver", "pso_mip_start"):
        data = valid[valid["solver_variant"] == strategy]
        fig.add_trace(go.Bar(name=strategy, x=list(metrics), y=[float(data[column].notna().mean()) if len(data) else 0.0 for column in metrics.values()]))
    fig.update_layout(title="Taxa de sucesso por métrica", template="plotly_white", barmode="group", yaxis=dict(range=[0, 1], tickformat=".0%"), yaxis_title="Taxa de sucesso")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Diretório out contendo shards telemetry")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output or Path(args.input) / "telemetry_consolidated")
    output.mkdir(parents=True, exist_ok=True)
    summary = read_shards(args.input, ".run_summary")
    iterations = read_shards(args.input, ".pso_iterations")
    events = read_shards(args.input, ".mip_events")
    if summary.empty:
        raise SystemExit("Nenhum shard run_summary encontrado")
    summary["completed_total_seconds"] = summary["total_seconds"].where(~summary["timed_out"].fillna(True))
    summary.to_parquet(output / "run_summary.parquet", index=False)
    if not iterations.empty: iterations.to_parquet(output / "pso_iterations.parquet", index=False)
    if not events.empty: events.to_parquet(output / "mip_events.parquet", index=False)
    valid, missing = pairs(summary)
    missing.to_json(output / "missing_pairs.json", orient="records", indent=2, default_handler=str)
    profile_figure(valid, "first_feasible_seconds", "Performance profile — primeira solução factível").write_html(output / "performance_profile_first_feasible.html", include_plotlyjs=True)
    profile_figure(valid, "gap_target_seconds", "Performance profile — gap-alvo").write_html(output / "performance_profile_gap_target.html", include_plotlyjs=True)
    profile_figure(valid, "completed_total_seconds", "Performance profile — execução concluída").write_html(output / "performance_profile_completed_total_time.html", include_plotlyjs=True)
    success_figure(valid).write_html(output / "success_rate_by_metric.html", include_plotlyjs=True)
    print(f"Telemetria consolidada em {output}; pares válidos={valid.groupby(['experiment_id', 'instance_file', 'weight', 'alpha'], dropna=False).ngroups}")

if __name__ == "__main__":
    main()
