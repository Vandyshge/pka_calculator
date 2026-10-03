from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import linregress

from ..thermodynamics.core import select_macro_microstates


def _selected_predictions(
    predictions: pd.DataFrame,
    *,
    selected_only: bool,
    value_column: str,
) -> pd.DataFrame:
    pred = predictions.copy()
    if selected_only and "selected" in pred.columns:
        pred = pred[pred["selected"]]
    required = {"molecule_id", value_column}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    return pred


def fit_linear_calibration(
    predictions: pd.DataFrame,
    experimental: pd.DataFrame,
    *,
    value_column: str = "value",
    experimental_column: str = "experimental",
    group_columns: Sequence[str] = ("level", "property"),
    selected_only: bool = True,
    min_points: int = 2,
) -> pd.DataFrame:
    """Fit a train-only linear correction ``experimental = a*calculated + b``.

    ``experimental`` should contain only the calibration/training molecules.  The
    returned coefficients are frozen and can then be applied to independent test
    predictions with :func:`apply_linear_calibration`.

    This direction is intentional: it produces a directly applicable correction
    from a raw quantum-chemical prediction to the experimental pK scale.
    """
    if min_points < 2:
        raise ValueError("min_points must be >= 2")

    pred = _selected_predictions(
        predictions,
        selected_only=selected_only,
        value_column=value_column,
    )
    if experimental_column not in experimental.columns:
        raise ValueError(
            f"experimental table must contain {experimental_column!r}"
        )
    if "molecule_id" not in experimental.columns:
        raise ValueError("experimental table must contain 'molecule_id'")

    merge_cols = ["molecule_id", experimental_column]
    merged = pred.merge(
        experimental[merge_cols],
        on="molecule_id",
        how="inner",
    ).dropna(subset=[value_column, experimental_column])

    groups = list(group_columns)
    missing_groups = [x for x in groups if x not in merged.columns]
    if missing_groups:
        raise ValueError(
            f"predictions missing calibration group columns: {missing_groups}"
        )

    rows: list[dict] = []
    for key, group in merged.groupby(groups, dropna=False):
        if len(group) < min_points:
            continue

        x = group[value_column].astype(float).to_numpy()
        y = group[experimental_column].astype(float).to_numpy()
        if np.allclose(x, x[0]):
            raise ValueError(
                f"Cannot fit linear calibration: calculated values are constant "
                f"for group={key!r}"
            )

        fit = linregress(x, y)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(groups, key_tuple)}
        row.update(
            {
                "slope": float(fit.slope),
                "intercept": float(fit.intercept),
                "r2_train": float(fit.rvalue**2),
                "stderr_slope": float(fit.stderr) if fit.stderr is not None else np.nan,
                "n_train": int(len(group)),
                "equation": "experimental = slope * calculated + intercept",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def apply_linear_calibration(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    value_column: str = "value",
    output_column: str = "value_linear_calibrated",
    group_columns: Sequence[str] = ("level", "property"),
    reselect_macro: bool = True,
) -> pd.DataFrame:
    """Apply frozen coefficients and reselect macro states on the new scale.

    Reselection is route-aware: acid pKa and pKb use the minimum calibrated
    microscopic value, while conjugate-acid pKaH uses the maximum.  This also
    handles a fitted negative slope, which reverses the ordering of sites.
    """
    pred = predictions.copy()
    groups = list(group_columns)
    required_cal = set(groups) | {"slope", "intercept"}
    missing_cal = required_cal - set(calibration.columns)
    if missing_cal:
        raise ValueError(f"calibration missing columns: {sorted(missing_cal)}")
    missing_pred = (set(groups) | {value_column}) - set(pred.columns)
    if missing_pred:
        raise ValueError(f"predictions missing columns: {sorted(missing_pred)}")

    coeff = calibration[groups + ["slope", "intercept"]].copy()
    if coeff.duplicated(groups).any():
        raise ValueError("calibration must contain one coefficient row per group")

    out = pred.merge(coeff, on=groups, how="left", validate="many_to_one")
    out[output_column] = (
        out["slope"].astype(float) * out[value_column].astype(float)
        + out["intercept"].astype(float)
    )
    if reselect_macro:
        out = select_macro_microstates(out, value_column=output_column)
    return out
