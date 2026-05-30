from pathlib import Path
import pandas as pd
from .utils import read_results_df


def gap_table_to_latex_alpha_weight(
    tabela_media: pd.DataFrame,
    clientes_order=None,
    classes_order=None,
    alpha_order=None,
    weight_order=None,
    class_roman_map=None,
    caption=None,
    label=None,
    table_env=True,
):
    tab = tabela_media.copy()

    if not isinstance(tab.index, pd.MultiIndex) or tab.index.nlevels != 2:
        raise ValueError(
            "tabela_media precisa ter índice MultiIndex: (clientes, classe)."
        )
    if not isinstance(tab.columns, pd.MultiIndex) or tab.columns.nlevels != 2:
        raise ValueError(
            "tabela_media precisa ter colunas MultiIndex: (alpha, weight_label)."
        )

    tab.index = tab.index.set_names(["clientes", "classe"])
    tab.columns = tab.columns.set_names(["alpha", "weight_label"])

    def num_key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))

    def weight_key(w):
        s = str(w)
        digits = "".join(ch for ch in s if ch.isdigit())
        return (0, int(digits)) if digits else (1, s)

    if clientes_order is None:
        clientes_order = sorted(
            tab.index.get_level_values("clientes").unique().tolist(), key=num_key
        )

    raw_classes = tab.index.get_level_values("classe").unique().tolist()
    has_all = any(str(c).lower() == "all" for c in raw_classes)

    if classes_order is None:
        sem_all = [c for c in raw_classes if str(c).lower() != "all"]
        classes_order = sorted(sem_all, key=num_key)
        if has_all:
            classes_order.append("all")
    else:
        classes_order = list(classes_order)
        if has_all and not any(str(c).lower() == "all" for c in classes_order):
            classes_order.append("all")

    if alpha_order is None:
        alpha_order = sorted(
            tab.columns.get_level_values("alpha").unique().tolist(), key=num_key
        )
    if weight_order is None:
        weight_order = sorted(
            tab.columns.get_level_values("weight_label").unique().tolist(),
            key=weight_key,
        )

    if class_roman_map is None:
        class_roman_map = {
            "1": "I",
            "2": "II",
            "3": "III",
            "4": "IV",
            1: "I",
            2: "II",
            3: "III",
            4: "IV",
        }

    full_rows = pd.MultiIndex.from_product(
        [clientes_order, classes_order], names=["clientes", "classe"]
    )
    full_cols = pd.MultiIndex.from_product(
        [alpha_order, weight_order], names=["alpha", "weight_label"]
    )
    mat = tab.reindex(index=full_rows, columns=full_cols)

    def alpha_label(a):
        try:
            return rf"$\alpha={float(a):g}$"
        except Exception:
            return rf"$\alpha={a}$"

    def class_label(cl):
        if str(cl).lower() == "all":
            return "All instances"
        return class_roman_map.get(cl, class_roman_map.get(str(cl), str(cl)))

    def clients_label(c):
        try:
            return f"{int(float(c))} clients"
        except Exception:
            return f"{c} clients"

    def fmt_pct(v):
        if pd.isna(v):
            return "-"
        return f"{v:.1f}".replace(".", ",") + r"\%"

    A = len(alpha_order)
    W = len(weight_order)
    colspec = "ll" + "c" * (A * W)

    lines = []
    if table_env:
        lines += [r"\begin{table}[t]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")

    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # Header 1: blocos por alpha
    row1 = [r"\multicolumn{1}{c}{\textbf{}}", r"\multicolumn{1}{c}{\textbf{}}"]
    row1 += [
        rf"\multicolumn{{{W}}}{{c}}{{\textbf{{{alpha_label(a)}}}}}" for a in alpha_order
    ]
    lines.append(" & ".join(row1) + r" \\")

    # cmidrule por alpha
    cmid = []
    start = 3
    for _ in alpha_order:
        end = start + W - 1
        cmid.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmid))

    # Header 2: pesos
    row2 = [
        r"\multicolumn{1}{c}{\textbf{Clientes}}",
        r"\multicolumn{1}{c}{\textbf{Classe}}",
    ]
    for _a in alpha_order:
        for w in weight_order:
            row2.append(rf"\multicolumn{{1}}{{c}}{{\textbf{{{w}}}}}")
    lines.append(" & ".join(row2) + r" \\")
    lines.append(r"\midrule")

    # Corpo: linhas por clientes e classe (incluindo all)
    for i, c in enumerate(clientes_order):
        rows_c = [(c, cl) for cl in classes_order]
        nrows = len(rows_c)

        for j, (_, cl) in enumerate(rows_c):
            row = (
                [rf"\multirow{{{nrows}}}{{*}}{{\textbf{{{clients_label(c)}}}}}"]
                if j == 0
                else [""]
            )
            row.append(class_label(cl))

            for a in alpha_order:
                for w in weight_order:
                    row.append(fmt_pct(mat.loc[(c, cl), (a, w)]))

            lines.append(" & ".join(row) + r" \\")

        if i != len(clientes_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if table_env:
        lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_gap_tables(out_dir: Path | str = "out") -> None:
    out_dir = Path(out_dir)
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    df = read_results_df()
    # ensure gap column exists and non-negative
    df["gap"] = (
        pd.to_numeric(df.get("gap", pd.NA), errors="coerce").fillna(0).clip(lower=0)
    )

    # prepare required columns
    base = df.copy()
    required_cols = ["instancia", "alpha", "clientes", "classe", "weight_label", "gap"]
    missing = [c for c in required_cols if c not in base.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    dados = base[required_cols].copy()
    dados["alpha"] = pd.to_numeric(dados["alpha"], errors="coerce")
    dados["clientes"] = pd.to_numeric(dados["clientes"], errors="coerce")
    dados["gap"] = pd.to_numeric(dados["gap"], errors="coerce")
    dados = dados.dropna(subset=["alpha", "clientes", "classe", "weight_label", "gap"])
    dados["clientes"] = dados["clientes"].astype(int)
    dados["classe"] = dados["classe"].astype(int).astype(str)
    dados["weight_label"] = dados["weight_label"].astype(str)

    # gap pct
    dados["gap_pct"] = dados["gap"] * 100

    # Remove extremos por grupo (IQR por weight_label)
    iqr_stats = (
        dados.groupby("weight_label")["gap_pct"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )
    iqr_stats["iqr"] = iqr_stats["q3"] - iqr_stats["q1"]
    iqr_stats["lim_inf"] = iqr_stats["q1"] - 1.5 * iqr_stats["iqr"]
    iqr_stats["lim_sup"] = iqr_stats["q3"] + 1.5 * iqr_stats["iqr"]

    dados_aux = dados.join(iqr_stats[["lim_inf", "lim_sup"]], on="weight_label")
    dados_sem_extremos = dados_aux[
        (dados_aux["gap_pct"] >= dados_aux["lim_inf"])
        & (dados_aux["gap_pct"] <= dados_aux["lim_sup"])
    ].drop(columns=["lim_inf", "lim_sup"])

    # resumo por alpha, clientes, classe, weight_label
    resumo = dados_sem_extremos.groupby(
        ["alpha", "clientes", "classe", "weight_label"], as_index=False
    ).agg(gap_medio_pct=("gap_pct", "mean"))

    # add All instances per clientes
    resumo_all = dados_sem_extremos.groupby(
        ["alpha", "clientes", "weight_label"], as_index=False
    ).agg(gap_medio_pct=("gap_pct", "mean"))
    resumo_all["classe"] = "all"

    resumo = pd.concat([resumo, resumo_all], ignore_index=True)

    # pivot table
    tabela_media = resumo.pivot_table(
        index=["clientes", "classe"],
        columns=["alpha", "weight_label"],
        values="gap_medio_pct",
        aggfunc="mean",
    )

    # determine orders
    alpha_order = sorted(tabela_media.columns.get_level_values(0).unique().tolist())
    weight_order = sorted(
        tabela_media.columns.get_level_values(1).unique().tolist(),
        key=lambda s: (
            int("".join(ch for ch in str(s) if ch.isdigit()))
            if any(ch.isdigit() for ch in str(s))
            else 10**9
        ),
    )
    clientes_order = sorted(tabela_media.index.get_level_values(0).unique().tolist())
    classes_order = sorted(
        [
            c
            for c in tabela_media.index.get_level_values(1).unique().tolist()
            if str(c).lower() != "all"
        ],
        key=lambda x: float(x),
    )
    if any(
        str(c).lower() == "all" for c in tabela_media.index.get_level_values(1).unique()
    ):
        classes_order.append("all")

    latex_code = gap_table_to_latex_alpha_weight(
        tabela_media,
        clientes_order=clientes_order,
        classes_order=classes_order,
        alpha_order=alpha_order,
        weight_order=weight_order,
        caption="Average gap (\\%) by clients/class (rows) and $\\alpha$/weight (columns), without outliers.",
        label="tab:gap_alpha_weight",
        table_env=True,
    )

    (latex_dir / "gap_alpha_weight.tex").write_text(latex_code)
