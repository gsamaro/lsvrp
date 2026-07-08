from pathlib import Path
import re
import pandas as pd
import numpy as np
from .utils import read_results_df


def _get_ct(df, by):
    key_cols = ["file", "weight_hash", "weight_label", "alpha", by]
    missing = [col for col in key_cols + ["f_sum_t"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for ct calculation: {missing}")

    df_f_sum_t = (
        pd.pivot_table(
            df,
            index=["file", "weight_hash", "weight_label", "alpha", by],
            values=["f_sum_t"],
            aggfunc="sum",
        )
        .reset_index()
    )
    df_f_sum_t_001 = df_f_sum_t.query("alpha == 0.01").drop(columns=["alpha"])
    df_f_sum_t_099 = df_f_sum_t.query("alpha == 0.99").drop(columns=["alpha"])
    df_ct = pd.merge(
        df_f_sum_t_001,
        df_f_sum_t_099,
        on=["file", "weight_hash", "weight_label", by],
        suffixes=["_001", "_099"],
    )
    if df_ct.empty:
        return pd.DataFrame(columns=["weight_label", by, "ct"])
    df_ct["ct"] = (
        np.abs(df_ct["f_sum_t_099"] - df_ct["f_sum_t_001"]) / df_ct["f_sum_t_001"]
    )
    return pd.pivot_table(
        df_ct, index=["weight_label", by], values="ct", aggfunc="mean"
    ).reset_index()


def df_to_latex_table(
    df: pd.DataFrame,
    caption=None,
    label=None,
    table_env=True,
    decimal_places=3,
    weights_in_columns=False,
    colname_map=None,
    font_size=r"\small",
):
    if colname_map is None:
        colname_map = {}

    def _header_label(col):
        name = colname_map.get(col, str(col))
        if ("$" not in name) and ("\\" not in name):
            name = name.replace("_", r"\_")
        return name

    if weights_in_columns:
        group_cols = [c for c in df.columns if c not in {"weight_label", "ct"}]
        if len(group_cols) != 1:
            raise ValueError(
                "weights_in_columns expects exactly one grouping column besides 'weight_label' and 'ct'."
            )
        group_col = group_cols[0]
        df_wide = pd.pivot_table(
            df, index=group_col, columns="weight_label", values="ct", aggfunc="mean"
        )

        # sort columns by w_{i}
        def _key(c):
            s = str(c)
            m = re.search(r"w_\{(\d+)\}", s)
            return int(m.group(1)) if m else 10**9

        ordered_cols = sorted(list(df_wide.columns), key=_key)
        df_wide = df_wide.reindex(columns=ordered_cols)

        lines = []
        if table_env:
            lines += [r"\begin{table}[htbp]", r"\centering"]
            if font_size:
                lines.append(font_size)
        lines.append(rf"\begin{{tabular}}{{l{ 'c'*len(df_wide.columns) }}}")
        lines.append(r"\toprule")
        ct_title = _header_label("ct")
        lines.append(
            rf"\multicolumn{{{len(df_wide.columns) + 1}}}{{c}}{{\textbf{{{ct_title}}}}} \\"
        )
        lines.append(r"\midrule")
        header = [r"\multicolumn{1}{c}{\textbf{}}"] + [
            rf"\multicolumn{{1}}{{c}}{{\textbf{{{str(w)}}}}}" for w in df_wide.columns
        ]
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")
        for idx_val, row in df_wide.iterrows():
            left = str(idx_val)
            out = [left]
            for w in df_wide.columns:
                val = row[w]
                if pd.isna(val):
                    out.append("")
                else:
                    out.append(f"{(val*100):.1f}\\%")
            lines.append(" & ".join(out) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        if table_env:
            lines.append(r"\end{table}")
        return "\n".join(lines)

    # fallback: plain table using pandas
    lines = []
    if table_env:
        lines += [r"\begin{table}[htbp]", r"\centering"]
        if font_size:
            lines.append(font_size)
    lines.append(
        df.to_latex(index=False, float_format=(lambda x: f"{x:.{decimal_places}f}"))
    )
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    if table_env:
        lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_sensitivity_tables(
    out_dir: Path | str = "out", font_size=r"\small"
) -> None:
    out_dir = Path(out_dir)
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    df = read_results_df()
    # prepare f_sum_t if missing
    if "f_sum_t" not in df.columns:
        df["f_sum_t"] = df[["f1", "f2", "f3", "f4", "f5"]].sum(axis=1)

    df_ct_clientes = _get_ct(df, "clientes")
    if df_ct_clientes.empty:
        (latex_dir / "sensitivity_ct_clientes.tex").write_text(
            "% Sensitivity table skipped: no matched rows for alpha 0.01 and 0.99.\n"
        )
    else:
        tex1 = df_to_latex_table(
            df_ct_clientes,
            caption="Sensitivity (ct) by number of clients",
            label="tab:sensitivity_ct_clientes",
            weights_in_columns=True,
            font_size=font_size,
            colname_map={
                "weight_label": "Weight",
                "clientes": "Clients",
                "ct": "$|CT^{(0.99)} - CT^{(0.01)}|/|CT^{(0.01)}|$",
            },
        )
        (latex_dir / "sensitivity_ct_clientes.tex").write_text(tex1)

    df_ct_classe = _get_ct(df, "classe")
    if df_ct_classe.empty:
        (latex_dir / "sensitivity_ct_classe.tex").write_text(
            "% Sensitivity table skipped: no matched rows for alpha 0.01 and 0.99.\n"
        )
    else:
        tex2 = df_to_latex_table(
            df_ct_classe,
            caption="Sensitivity (ct) by class",
            label="tab:sensitivity_ct_classe",
            weights_in_columns=True,
            font_size=font_size,
            colname_map={
                "weight_label": "Weight",
                "clientes": "Clients",
                "ct": "$|CT^{(0.99)} - CT^{(0.01)}|/|CT^{(0.01)}|$",
            },
        )
        (latex_dir / "sensitivity_ct_classe.tex").write_text(tex2)
