from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


def metrics_table(
    predictions: pd.DataFrame,
    experimental: pd.DataFrame,
    *,
    selected_only: bool = True,
) -> pd.DataFrame:
    pred = predictions.copy()
    if selected_only and "selected" in pred.columns:
        pred = pred[pred["selected"]]
    merged = pred.merge(experimental, on="molecule_id", how="inner").dropna(
        subset=["value", "experimental"]
    )
    rows = []
    for (level, prop), group in merged.groupby(["level", "property"]):
        y = group["value"].to_numpy(float)
        x = group["experimental"].to_numpy(float)
        error = y - x
        row = {
            "level": level,
            "property": prop,
            "n": len(group),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
            "r2": float(np.corrcoef(x, y)[0, 1] ** 2) if len(group) > 1 else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_fit(
    predictions: pd.DataFrame,
    experimental: pd.DataFrame,
    *,
    selected_only: bool = True,
) -> pd.DataFrame:
    """Legacy diagnostic fit ``calculated = slope*experimental + intercept``.

    For a correction that can be fit on train and directly applied to test, use
    :func:`pka_calculator.analysis.fit_linear_calibration` instead.
    """
    pred = predictions.copy()
    if selected_only and "selected" in pred.columns:
        pred = pred[pred["selected"]]
    merged = pred.merge(experimental, on="molecule_id", how="inner").dropna(
        subset=["value", "experimental"]
    )
    rows = []
    for (level, prop), group in merged.groupby(["level", "property"]):
        if len(group) < 2:
            continue
        fit = linregress(group["experimental"].astype(float), group["value"].astype(float))
        rows.append(
            {
                "level": level,
                "property": prop,
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r2": float(fit.rvalue**2),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)
