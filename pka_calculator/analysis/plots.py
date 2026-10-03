from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_predictions(
    predictions: pd.DataFrame,
    experimental: pd.DataFrame,
    output_dir: str | Path,
    *,
    selected_only: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = predictions.copy()
    if selected_only and "selected" in pred.columns:
        pred = pred[pred["selected"]]
    merged = pred.merge(experimental, on="molecule_id", how="inner").dropna(
        subset=["value", "experimental"]
    )
    paths = []
    for (level, prop), group in merged.groupby(["level", "property"]):
        x = group["experimental"].to_numpy(float)
        y = group["value"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y)
        lo = float(min(x.min(), y.min()) - 0.5)
        hi = float(max(x.max(), y.max()) + 0.5)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Experimental {prop}")
        ax.set_ylabel(f"Calculated {prop}")
        ax.set_title(str(level))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"{level}_{prop}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
