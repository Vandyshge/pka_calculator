import pandas as pd
import pytest

from pka_calculator.constants import HARTREE_TO_KCAL_MOL
from pka_calculator.geometry import select_conformers_within_energy_window


def test_select_conformers_within_window_and_cap():
    one_kcal_eh = 1.0 / HARTREE_TO_KCAL_MOL
    results = pd.DataFrame(
        {
            "molecule_id": ["A", "A", "A", "A", "B", "B"],
            "level": ["L"] * 6,
            "state_id": ["c1", "c2", "c3", "c4", "c1", "c2"],
            "final_single_point_energy_eh": [
                -100.0,
                -100.0 + one_kcal_eh,
                -100.0 + 2.5 * one_kcal_eh,
                -100.0 + 4.0 * one_kcal_eh,
                -50.0,
                -50.0 + 2.0 * one_kcal_eh,
            ],
            "valid": [True, True, True, True, True, False],
        }
    )

    selected = select_conformers_within_energy_window(
        results,
        energy_window_kcal_mol=3.0,
        max_conformers_per_molecule=2,
    )

    assert selected.groupby("molecule_id")["state_id"].apply(list).to_dict() == {
        "A": ["c1", "c2"],
        "B": ["c1"],
    }
    assert selected.groupby("molecule_id")["conformer_energy_rank"].apply(list).to_dict() == {
        "A": [1, 2],
        "B": [1],
    }
    assert selected.loc[selected["molecule_id"] == "A", "relative_energy_kcal_mol"].tolist() == pytest.approx([0.0, 1.0])


def test_select_conformers_window_zero_keeps_energy_minimum():
    results = pd.DataFrame(
        {
            "molecule_id": ["A", "A"],
            "state_id": ["higher", "minimum"],
            "final_single_point_energy_eh": [-9.9, -10.0],
            "valid": [True, True],
        }
    )

    selected = select_conformers_within_energy_window(
        results,
        energy_window_kcal_mol=0.0,
        max_conformers_per_molecule=None,
    )

    assert selected["state_id"].tolist() == ["minimum"]


@pytest.mark.parametrize(
    ("window", "cap"),
    [(-1.0, 2), (3.0, 0)],
)
def test_select_conformers_rejects_invalid_limits(window, cap):
    results = pd.DataFrame(
        {"molecule_id": ["A"], "final_single_point_energy_eh": [-10.0]}
    )
    with pytest.raises(ValueError):
        select_conformers_within_energy_window(
            results,
            energy_window_kcal_mol=window,
            max_conformers_per_molecule=cap,
        )
