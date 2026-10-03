from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_interactive_html(
    predictions: pd.DataFrame,
    experimental: pd.DataFrame,
    output_file: str | Path,
    *,
    selected_only: bool = True,
) -> Path:
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pka-calculator[interactive] to generate HTML reports") from exc

    pred = predictions.copy()
    if selected_only and "selected" in pred.columns:
        pred = pred[pred["selected"]]
    merged = pred.merge(experimental, on="molecule_id", how="inner")
    fig = px.scatter(
        merged,
        x="experimental",
        y="value",
        color="level",
        symbol="property",
        hover_data=[x for x in ["molecule_id", "state_id", "site_id", "route"] if x in merged],
        labels={"experimental": "Experimental", "value": "Calculated"},
        title="Experimental vs calculated acid/base constants",
    )
    if not merged.empty:
        lo = min(merged["experimental"].min(), merged["value"].min())
        hi = max(merged["experimental"].max(), merged["value"].max())
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line={"dash": "dash"})
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, output_file, include_plotlyjs="cdn", full_html=True)
    return output_file
