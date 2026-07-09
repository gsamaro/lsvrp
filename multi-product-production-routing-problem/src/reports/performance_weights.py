import ast
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from .utils import read_results_df


def _weight_key(label) -> int:
    s = str(label)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 10**9


def _find_chrome_executable() -> str | None:
    chrome_for_testing = (
        Path.home()
        / "Library/Application Support/choreographer/deps/chrome-mac-arm64"
        / "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    )
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chrome"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        chrome_for_testing,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _write_plotly_png_with_chrome(fig, html_path: Path, png_path: Path) -> None:
    chrome = _find_chrome_executable()
    if chrome is None:
        raise RuntimeError("Chrome executable not found for Plotly PNG fallback.")

    fig.write_html(str(html_path), include_plotlyjs="inline", full_html=True)
    width = fig.layout.width or 1100
    height = fig.layout.height or 700
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--virtual-time-budget=3000",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wins_table_to_latex(
    win_counts: pd.DataFrame,
    clientes_order=None,
    classes_order=None,
    weight_order=None,
    class_roman_map=None,
    caption=None,
    label=None,
    table_env=True,
    font_size=r"\small",
):
    wc = win_counts.copy()
    wc["clientes"] = pd.to_numeric(wc["clientes"], errors="ignore")
    wc["classe"] = pd.to_numeric(wc["classe"], errors="ignore")

    if clientes_order is None:
        clientes_order = sorted(pd.unique(wc["clientes"]).tolist())
    if classes_order is None:
        classes_order = sorted(pd.unique(wc["classe"]).tolist())
    if weight_order is None:
        weight_order = sorted(pd.unique(wc["weight_label"]).tolist())

    if class_roman_map is None:
        class_roman_map = {1: "I", 2: "II", 3: "III", 4: "IV"}

    mat = wc.pivot_table(
        index=["clientes", "classe"],
        columns="weight_label",
        values="win_count",
        aggfunc="sum",
        fill_value=0,
    )

    # Reindex rows and columns
    full_rows = pd.MultiIndex.from_product(
        [clientes_order, classes_order], names=["clientes", "classe"]
    )
    mat = mat.reindex(index=full_rows, columns=weight_order, fill_value=0)

    # Add All instances per clientes
    all_rows = []
    for c in clientes_order:
        row_sum = mat.loc[c].sum(axis=0)
        all_rows.append(((c, "all"), row_sum))
    all_df = pd.DataFrame(
        [r[1] for r in all_rows],
        index=pd.MultiIndex.from_tuples(
            [r[0] for r in all_rows], names=["clientes", "classe"]
        ),
    )
    mat_with_all = pd.concat([mat, all_df]).sort_index()

    total_wins = mat_with_all.sum(axis=0)
    grand_total = total_wins.sum()
    proportions = (total_wins / grand_total) if grand_total > 0 else total_wins * 0

    num_weights = len(weight_order)
    colspec = "l" + "l" + "c" * num_weights

    def class_label(cl):
        if str(cl).lower() == "all":
            return "All instances"
        return class_roman_map.get(cl, str(cl))

    lines = []
    if table_env:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if font_size:
            lines.append(font_size)

    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(
        rf"\multicolumn{{{num_weights + 2}}}{{c}}{{\# the best performing weight}} \\"
    )
    lines.append(r"\midrule")
    header = [r"\multicolumn{1}{c}{\textbf{}}", r"\multicolumn{1}{c}{\textbf{}}"] + [
        rf"\multicolumn{{1}}{{c}}{{\textbf{{{w}}}}}" for w in weight_order
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for idx, (c, cl) in enumerate(mat_with_all.index):
        # insert midrule between different clientes
        if idx > 0 and c != mat_with_all.index[idx - 1][0]:
            lines.append(r"\midrule")

        if cl == "all":
            row_label = class_label(cl)
        else:
            row_label = class_label(cl)

        if cl == mat_with_all.index[mat_with_all.index.get_loc((c, cl))][1]:
            pass

        # multirow for clientes
        # detect first occurrence of this clientes in index
        first_for_client = (
            mat_with_all.index.get_loc((c, mat_with_all.loc[c].index[0]))
            if False
            else None
        )

        # Build row
        # Use multirow only at first class row for each client
        # Simpler: use explicit \multirow with count of classes+1 (all)
        # We'll detect when cl is the first class for this client
        # Find classes for this client
        classes_for_client = [t[1] for t in mat_with_all.index if t[0] == c]
        num_rows = len(classes_for_client)
        # if this is the first class occurrence for this client, add multirow
        is_first = False
        # determine position of (c,cl) among client's rows
        positions = [i for i, t in enumerate(mat_with_all.index) if t[0] == c]
        if positions and positions[0] == idx:
            is_first = True

        if is_first:
            left = rf"\multirow{{{num_rows}}}{{*}}{{\textbf{{{int(c)} clients}}}}"
        else:
            left = ""

        row = [left, class_label(cl)]
        for w in weight_order:
            val = mat_with_all.loc[(c, cl), w]
            row.append(str(int(val)) if not pd.isna(val) else "0")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")
    prop_row = [rf"\multicolumn{{2}}{{l}}{{\textbf{{Proportion}}}}"] + [
        f"{proportions[w]:.2f}" for w in weight_order
    ]
    lines.append(" & ".join(prop_row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    if table_env:
        lines.append(r"\end{table}")

    return "\n".join(lines)


def _winners2_to_latex_alpha_cols_tau_rows(
    winners2: pd.DataFrame,
    taus_order=None,
    alpha_order=None,
    caption=None,
    label=None,
    table_env=True,
    font_size=r"\small",
):
    w = winners2.copy()
    w["alpha"] = pd.to_numeric(w["alpha"], errors="coerce")
    w["tau"] = pd.to_numeric(w["tau"], errors="coerce")

    if alpha_order is None:
        alpha_order = sorted(pd.unique(w["alpha"]).tolist())
    if taus_order is None:
        taus_order = sorted(pd.unique(w["tau"]).tolist())

    mat = w.pivot_table(
        index="tau", columns="alpha", values="weight_label", aggfunc="first"
    ).reindex(index=taus_order, columns=alpha_order)

    colspec = "l" + "c" * len(alpha_order)

    def alpha_label(a):
        return rf"$\alpha={a:g}$"

    def tau_label(t):
        return rf"$\rho(\tau)={t:g}$"

    lines = []
    if table_env:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if font_size:
            lines.append(font_size)

    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{1}{c}{\textbf{}} & "
        + " & ".join(
            [
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{alpha_label(a)}}}}}"
                for a in alpha_order
            ]
        )
        + r" \\"
    )
    lines.append(r"\midrule")

    for tau in mat.index.tolist():
        row = [tau_label(tau)]
        for a in alpha_order:
            v = mat.loc[tau, a]
            row.append("" if pd.isna(v) else str(v))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    if table_env:
        lines.append(r"\end{table}")
    return "\n".join(lines)


def _count_wins_by_clientes_classe(
    winners: pd.DataFrame,
    clientes_order=None,
    classes_order=None,
):
    """
    Conta quantas vezes cada weight_label venceu para cada combinação de clientes e classe,
    agregando sobre todos os valores de tau.
    """
    w = winners.copy()
    w["clientes"] = pd.to_numeric(w["clientes"], errors="ignore")
    w["classe"] = pd.to_numeric(w["classe"], errors="ignore")

    if clientes_order is None:
        clientes_order = sorted(pd.unique(w["clientes"]).tolist())
    if classes_order is None:
        classes_order = sorted(pd.unique(w["classe"]).tolist())

    # Agregar contagens por (clientes, classe, weight_label) - soma sobre tau
    win_counts = (
        w.groupby(["clientes", "classe", "weight_label"], as_index=False)
        .size()
        .rename(columns={"size": "win_count"})
    )

    return win_counts


def _weight_mapping_to_latex(
    label_to_vector: dict,
    caption=None,
    label=None,
    table_env=True,
    decimal_places=3,
    font_size=r"\small",
):
    """
    Gera tabela LaTeX mostrando o mapeamento entre weight labels e seus vetores.
    Linhas: weight labels (w_1, w_2, ...)
    Colunas: componentes do vetor (v_1, v_2, v_3, v_4, v_5)
    """

    # Ordenar labels
    def _weight_key(wlabel):
        s = str(wlabel)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 10**9

    sorted_labels = sorted(label_to_vector.keys(), key=_weight_key)

    # Número de componentes (assumindo que todos os vetores têm o mesmo tamanho)
    num_components = len(list(label_to_vector.values())[0])

    # Construir LaTeX
    colspec = "l" + "c" * num_components

    lines = []
    if table_env:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if font_size:
            lines.append(font_size)

    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # Header: componentes do vetor
    header = [r"\multicolumn{1}{c}{\textbf{}}"] + [
        rf"\multicolumn{{1}}{{c}}{{\textbf{{$v_{{{i}}}$}}}}"
        for i in range(1, num_components + 1)
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Corpo: cada linha é um weight label com seus valores arredondados
    for wlabel in sorted_labels:
        vector = label_to_vector[wlabel]
        row = [wlabel] + [f"{v:.{decimal_places}f}" for v in vector]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    if table_env:
        lines.append(r"\end{table}")

    return "\n".join(lines)


def _generate_performance_profile_figure(
    out_data: pd.DataFrame, figs_dir: Path, stem: str = "performance_profile_weights"
) -> None:
    """
    Gera figura Plotly de performance profile (célula 24 do notebook).

    Parâmetros:
    - out_data: DataFrame com colunas [weight_label, alpha, tau, count, count_norm]
    - figs_dir: diretório para salvar a figura
    - stem: nome base (sem extensão) para salvar html/png/svg
    """
    out_plot = out_data.copy()
    # Mapeamento de subscritos Unicode para exibição
    subscript_map = {
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
    }

    # Converter labels LaTeX para formato com subscritos visíveis (w₁, w₂, etc)
    def convert_weight_label(label):
        if isinstance(label, str) and label.startswith("$w_") and label.endswith("$"):
            num = label[4:-2]  # extrai número entre $w_{ e }$
            return "w" + subscript_map.get(num, num)
        return label

    out_plot["weight_label_display"] = out_plot["weight_label"].apply(
        convert_weight_label
    )

    fig = px.line(
        out_plot[["tau", "count_norm", "weight_label_display", "alpha"]],
        x="tau",
        y="count_norm",
        color="weight_label_display",
        facet_row="alpha",
        height=700,
        width=1100,
        markers=False,
        labels={
            "count_norm": "Proporção",
            "weight_label_display": "Peso",
        },
    )

    fig.update_xaxes(range=[1, None])

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman, serif", size=12),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            title="Peso",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(r=340),
    )

    for i in range(1, 10):
        fig.update_xaxes(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=False,
            row=i if i <= 5 else None,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=False,
            row=i if i <= 5 else None,
        )

    png_path = figs_dir / f"{stem}.png"
    svg_path = figs_dir / f"{stem}.svg"
    html_path = figs_dir / f"{stem}.html"
    try:
        fig.write_image(str(png_path))
        fig.write_image(str(svg_path))
    except Exception:
        _write_plotly_png_with_chrome(fig, html_path, png_path)


def _winners_by_tau_to_latex(
    winners: pd.DataFrame,
    taus_order=None,
    alpha_order=None,
    clientes_order=None,
    classes_order=None,
    class_roman_map=None,
    caption=None,
    label=None,
    table_env=True,
    font_size=r"\small",
) -> str:
    """
    Gera a tabela da célula 27: peso vencedor por tau, alpha, clientes e classe.
    """
    w = winners.copy()
    w["alpha"] = pd.to_numeric(w["alpha"], errors="coerce")
    w["tau"] = pd.to_numeric(w["tau"], errors="coerce")
    w["clientes"] = pd.to_numeric(w["clientes"], errors="ignore")
    w["classe"] = pd.to_numeric(w["classe"], errors="ignore")

    if alpha_order is None:
        alpha_order = sorted(pd.unique(w["alpha"]).tolist())
    if clientes_order is None:
        clientes_order = sorted(pd.unique(w["clientes"]).tolist())
    if classes_order is None:
        classes_order = sorted(pd.unique(w["classe"]).tolist())
    if taus_order is None:
        taus_order = sorted(pd.unique(w["tau"]).tolist())

    if class_roman_map is None:
        class_roman_map = {1: "I", 2: "II", 3: "III", 4: "IV"}

    rows = []
    for tau in taus_order:
        tau_rows = w[np.isclose(w["tau"], tau)]
        if tau_rows.empty:
            continue
        tau_rows = tau_rows.copy()
        tau_rows["tau_label"] = tau
        rows.append(tau_rows)
    if not rows:
        raise ValueError("Sem vencedores para os valores de tau escolhidos.")

    w = pd.concat(rows, ignore_index=True)
    mat = w.pivot_table(
        index="tau_label",
        columns=["alpha", "clientes", "classe"],
        values="weight_label",
        aggfunc="first",
    )
    full_cols = pd.MultiIndex.from_product(
        [alpha_order, clientes_order, classes_order],
        names=["alpha", "clientes", "classe"],
    )
    mat = mat.reindex(columns=full_cols)
    mat = mat.reindex(index=taus_order)

    if mat.dropna(how="all").empty:
        raise ValueError("Sem dados para montar wins_by_tau.tex.")

    A = len(alpha_order)
    C = len(clientes_order)
    K = len(classes_order)

    colspec = "l" + "c" * (A * C * K)

    def alpha_label(a):
        return rf"$\alpha={a:g}$"

    def class_label(cl):
        return class_roman_map.get(cl, str(cl))

    def tau_label(t):
        return rf"$\rho(\tau)={t:g}$"

    def weight_label(value):
        return "" if pd.isna(value) else str(value)

    lines = []
    if table_env:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if font_size:
            lines.append(font_size)

    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    lines.append(
        r"\multicolumn{1}{c}{\textbf{}} & "
        + " & ".join(
            [
                rf"\multicolumn{{{C*K}}}{{c}}{{\textbf{{{alpha_label(a)}}}}}"
                for a in alpha_order
            ]
        )
        + r" \\"
    )

    cmid = []
    start = 2
    for _ in alpha_order:
        end = start + (C * K) - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmid))

    row2 = [r"\multicolumn{1}{c}{\textbf{}}"]
    for _a in alpha_order:
        for c_val in clientes_order:
            row2.append(
                rf"\multicolumn{{{K}}}{{c}}{{\textbf{{clientes {int(c_val)}}}}}"
            )
    lines.append(" & ".join(row2) + r" \\")

    cmid2 = []
    start = 2
    for _a in alpha_order:
        for _c in clientes_order:
            end = start + K - 1
            cmid2.append(rf"\cmidrule(lr){{{start}-{end}}}")
            start = end + 1
    lines.append(" ".join(cmid2))

    row3 = [r"\multicolumn{1}{c}{\textbf{}}"]
    for _a in alpha_order:
        for _c in clientes_order:
            for cl in classes_order:
                row3.append(rf"\multicolumn{{1}}{{c}}{{\textbf{{{class_label(cl)}}}}}")
    lines.append(" & ".join(row3) + r" \\")
    lines.append(r"\midrule")

    for tau in mat.index.tolist():
        row = [tau_label(tau)]
        for a in alpha_order:
            for c_val in clientes_order:
                for cl in classes_order:
                    row.append(weight_label(mat.loc[tau, (a, c_val, cl)]))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    if table_env:
        lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_performance_weights(
    out_dir: str | Path = "out", font_size=r"\small"
) -> None:
    out_dir = Path(out_dir)
    latex_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    latex_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = read_results_df()

    # Save weight mapping (human-readable)
    if (
        "weight_hash" in df.columns
        and "weight_label" in df.columns
        and "weight" in df.columns
    ):
        hash_map = {
            h: r
            for h, r in zip(df["weight_hash"].unique(), df["weight_label"].unique())
        }
        mapping_lines = ["Hash map current:"]
        for h, w in hash_map.items():
            mapping_lines.append(
                f"{w}: {df.loc[df['weight_hash']==h, 'weight'].iloc[0]}"
            )
        (latex_dir / "weight_mapping.txt").write_text("\n".join(mapping_lines))

    # (Cell 24 + 27 + 36) logic relies on ratios (needs p?/f?_target) and grouping cols.
    f_cols = [f"p{i}" for i in range(1, 6)]
    target_cols = [f"f{i}_target" for i in range(1, 6)]
    required = f_cols + target_cols + ["alpha", "weight_label", "clientes", "classe"]
    if not all(c in df.columns for c in required):
        return

    for i in range(1, 6):
        p_col = f"p{i}"
        t_col = f"f{i}_target"
        out_col = f"desv_rel_targ_{i}"
        df[out_col] = df[p_col] / df[t_col].replace({0: np.nan})

    df_min_desv_rel = pd.pivot_table(
        df,
        index=["file", "time"],
        aggfunc={k: "min" for k in [f"desv_rel_targ_{i}" for i in range(1, 6)]},
    ).rename(
        columns={f"desv_rel_targ_{i}": f"min_desv_rel_targ_{i}" for i in range(1, 6)}
    )

    df_min = (
        pd.merge(df, df_min_desv_rel, on=["file", "time"])
        if not df_min_desv_rel.empty
        else df.copy()
    )

    for i in range(1, 6):
        df_min[f"ratio_{i}"] = df_min[f"desv_rel_targ_{i}"] / df_min.get(
            f"min_desv_rel_targ_{i}", df_min[f"desv_rel_targ_{i}"]
        )
    df_min[[f"ratio_{i}" for i in range(1, 6)]] = df_min[
        [f"ratio_{i}" for i in range(1, 6)]
    ].fillna(1)
    df_min_clean = df_min[df_min["ratio_1"] < 4]

    taus = np.arange(1, 4, 0.001)
    ratio_cols = [f"ratio_{i}" for i in range(1, 6)]
    df_long = df_min_clean.melt(
        id_vars=["weight_label", "alpha", "clientes", "classe"],
        value_vars=ratio_cols,
        var_name="ratio",
        value_name="ratio_value",
    ).dropna(subset=["ratio_value"])

    counts = (
        df_long.groupby(["alpha", "clientes", "classe", "weight_label"])["ratio_value"]
        .apply(lambda s: pd.Series({tau: (s <= tau).sum() for tau in taus}))
        .rename("count")
        .reset_index()
        .rename(columns={"level_4": "tau"})
    )

    winners = (
        counts.sort_values(
            ["alpha", "clientes", "classe", "tau", "count", "weight_label"]
        )
        .groupby(["alpha", "clientes", "classe", "tau"], as_index=False)
        .tail(1)
    )

    # (Cell 24) figura performance profile (agregada por alpha)
    out_for_fig = (
        df_long.groupby(["weight_label", "alpha"])["ratio_value"]
        .apply(lambda s: pd.Series({tau: (s <= tau).sum() for tau in taus}))
        .rename("count")
        .reset_index()
        .rename(columns={"level_2": "tau"})
    )
    if not out_for_fig.empty:
        out_for_fig["count_norm"] = out_for_fig["count"] / out_for_fig["count"].max()
        _generate_performance_profile_figure(out_for_fig, figs_dir)

    # (Cell 27) tabela dos pesos vencedores por tau, alpha, clientes e classe
    taus_of_interest = [1.0, 1.5]
    winners_filtered = winners[
        np.logical_or.reduce(
            [np.isclose(winners["tau"], tau) for tau in taus_of_interest]
        )
    ].copy()
    if not winners_filtered.empty:
        alpha_order_tau = sorted(winners_filtered["alpha"].unique())
        clientes_order_tau = sorted(winners_filtered["clientes"].unique())
        classes_order_tau = sorted(winners_filtered["classe"].unique())
        tex_counts = _winners_by_tau_to_latex(
            winners_filtered,
            taus_order=taus_of_interest,
            alpha_order=alpha_order_tau,
            clientes_order=clientes_order_tau,
            classes_order=classes_order_tau,
            caption="Peso Vencedor",
            label="tab:winner",
            table_env=True,
            font_size=font_size,
        )
        (latex_dir / "wins_by_tau.tex").write_text(tex_counts)

        # (Cell 35) vitórias agregadas a partir da tabela da célula 27
        win_counts = _count_wins_by_clientes_classe(
            winners_filtered,
            clientes_order=clientes_order_tau,
            classes_order=classes_order_tau,
        )
        weight_order = sorted(df["weight_label"].dropna().unique(), key=_weight_key)
        tex = _wins_table_to_latex(
            win_counts,
            clientes_order=clientes_order_tau,
            classes_order=classes_order_tau,
            weight_order=weight_order,
            caption="Global performance of a portion of weight space $W$",
            label="tab:weight_performance",
            table_env=True,
            font_size=font_size,
        )
        (latex_dir / "weight_performance.tex").write_text(tex)

    # Extra: tabela do melhor peso por alpha/tau (útil para checagem)
    agg = counts.groupby(["alpha", "tau", "weight_label"], as_index=False)[
        "count"
    ].sum()
    winners2 = (
        agg.sort_values(["alpha", "tau", "count", "weight_label"])
        .groupby(["alpha", "tau"], as_index=False)
        .tail(1)
    )
    if not winners2.empty:
        tex2 = _winners2_to_latex_alpha_cols_tau_rows(
            winners2,
            taus_order=taus_of_interest,
            alpha_order=sorted(winners2["alpha"].unique().tolist()),
            caption="Best weight per $\\alpha$ and $\\tau$",
            label="tab:winners_alpha_tau",
            table_env=True,
            font_size=font_size,
        )
        (latex_dir / "winners_alpha_tau.tex").write_text(tex2)

    # Weight mapping table in LaTeX (optional)
    if all(c in df.columns for c in ["weight_hash", "weight", "weight_label"]):
        weight_mapping = df[["weight_hash", "weight", "weight_label"]].drop_duplicates()
        label_to_vector: dict[str, list[float]] = {}
        for _, row in weight_mapping.iterrows():
            label = row["weight_label"]
            try:
                vector = ast.literal_eval(row["weight"])
                if isinstance(vector, (list, tuple)) and vector:
                    label_to_vector[str(label)] = list(vector)
            except (ValueError, SyntaxError):
                pass

        if label_to_vector:
            tex_weights = _weight_mapping_to_latex(
                label_to_vector,
                caption="Weight vector mapping",
                label="tab:weight_mapping",
                table_env=True,
                decimal_places=3,
                font_size=font_size,
            )
            (latex_dir / "weight_mapping.tex").write_text(tex_weights)
