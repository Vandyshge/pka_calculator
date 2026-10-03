import numpy as np
import pandas as pd

from pka_calculator.analysis import apply_linear_calibration, fit_linear_calibration


def test_train_fit_and_frozen_test_application():
    train_pred = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d"],
            "level": ["L"] * 4,
            "property": ["pKa"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
            "selected": [True] * 4,
        }
    )
    # exact relation exp = 1.5 * calc - 0.25
    train_exp = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d"],
            "experimental": [1.25, 2.75, 4.25, 5.75],
        }
    )
    fit = fit_linear_calibration(train_pred, train_exp)
    assert len(fit) == 1
    assert np.isclose(fit.loc[0, "slope"], 1.5)
    assert np.isclose(fit.loc[0, "intercept"], -0.25)

    test_pred = pd.DataFrame(
        {
            "molecule_id": ["test"],
            "level": ["L"],
            "property": ["pKa"],
            "value": [5.0],
            "selected": [True],
        }
    )
    out = apply_linear_calibration(test_pred, fit)
    assert np.isclose(out.loc[0, "value_linear_calibrated"], 7.25)


def test_linear_calibration_reselects_conjugate_acid_maximum_after_negative_slope():
    predictions = pd.DataFrame(
        {
            "molecule_id": ["ethanolamine", "ethanolamine"],
            "level": ["L", "L"],
            "route": ["conjugate_acid", "conjugate_acid"],
            "property": ["pKaH", "pKaH"],
            "state_id": ["protonated_N", "protonated_O"],
            "value": [2.0, 5.0],
            "selected": [False, True],
        }
    )
    calibration = pd.DataFrame(
        {"level": ["L"], "property": ["pKaH"], "slope": [-1.0], "intercept": [10.0]}
    )

    out = apply_linear_calibration(predictions, calibration)
    selected = out[out["selected"]]

    assert selected.iloc[0]["state_id"] == "protonated_N"
    assert selected.iloc[0]["value_linear_calibrated"] == out["value_linear_calibrated"].max()


def test_linear_calibration_reselects_acid_minimum_after_negative_slope():
    predictions = pd.DataFrame(
        {
            "molecule_id": ["acid", "acid"],
            "level": ["L", "L"],
            "route": ["acid", "acid"],
            "property": ["pKa", "pKa"],
            "state_id": ["deprotonated_1", "deprotonated_2"],
            "value": [2.0, 5.0],
            "selected": [True, False],
        }
    )
    calibration = pd.DataFrame(
        {"level": ["L"], "property": ["pKa"], "slope": [-1.0], "intercept": [10.0]}
    )

    out = apply_linear_calibration(predictions, calibration)
    selected = out[out["selected"]]

    assert selected.iloc[0]["state_id"] == "deprotonated_2"
    assert selected.iloc[0]["value_linear_calibrated"] == out["value_linear_calibrated"].min()
