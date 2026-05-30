from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import plotly.express as px
from .utils import read_results_df


def generate_performance_fob(out_dir: Union[str, Path] = "out") -> None:
    out_dir = Path(out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    df = read_results_df()

    omegas = np.arange(0, 4, 0.001)
    fob_cols = [f"desv_rel_targ_{i}" for i in range(1, 6)]

    df_long = df.melt(value_vars=fob_cols, var_name="fob", value_name="rtd_value")

    out = (
        df_long.groupby(["fob"])["rtd_value"]
        .apply(lambda s: pd.Series({omega: (s <= omega).sum() for omega in omegas}))
        .rename("count")
        .reset_index()
        .rename(columns={"level_1": "omega"})
    )

    out["count_norm"] = out["count"] / out["count"].max()

    out_plot = out.copy()
    out_plot["fob_label"] = (
        out_plot["fob"].str.extract(r"(\d+)")[0].apply(lambda x: f"f^{x}")
    )

    fig = px.line(
        out_plot[["omega", "count_norm", "fob_label"]],
        x="omega",
        y="count_norm",
        color="fob_label",
        height=600,
        markers=False,
    )

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", legend=dict(title="")
    )

    fig.update_xaxes(title="RTD limit")
    fig.update_yaxes(title="RTD proportion")

    # save images (requires kaleido)
    png_path = figs_dir / "performance_fob.png"
    svg_path = figs_dir / "performance_fob.svg"
    try:
        fig.write_image(str(png_path))
        fig.write_image(str(svg_path))
    except Exception:
        # fallback: open in browser if image export not available
        fig.write_html(str(figs_dir / "performance_fob.html"))
