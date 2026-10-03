import pandas as pd

from pka_calculator.constants import HARTREE_TO_KJ_MOL, rtln10_kj_mol
from pka_calculator.thermodynamics.core import (
    calibrate_reference,
    predict_acid_base,
    reaction_energy_difference,
    reaction_energy_table,
    select_macro_microstates,
)


def results_for_acid(reference: float, pka: float = 5.0):
    gn = -100.0
    factor = rtln10_kj_mol(298.15)
    gd_kj = gn * HARTREE_TO_KJ_MOL + pka * factor - reference
    gd = gd_kj / HARTREE_TO_KJ_MOL
    return pd.DataFrame(
        [
            {"molecule_id": "A", "level": "L", "state": "neutral", "state_id": "n", "site_id": None, "gibbs_free_energy_eh": gn, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "A", "level": "L", "state": "deprotonated", "state_id": "d1", "site_id": "O1", "gibbs_free_energy_eh": gd, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )


def test_calibrate_and_predict_known_proton_reference():
    known = -1104.5
    results = results_for_acid(known, pka=5.0)
    exp = pd.DataFrame({"molecule_id": ["A"], "experimental": [5.0]})
    refs, details = calibrate_reference(results, exp, route="acid")
    assert abs(refs.loc[0, "value_kj_mol"] - known) < 1e-9
    pred = predict_acid_base(results, refs, route="acid")
    assert abs(pred.loc[0, "value"] - 5.0) < 1e-9
    assert bool(pred.loc[0, "selected"])


def test_reaction_energy_difference_follows_route_direction():
    neutral = -100.0
    deprotonated = -99.9
    protonated = -100.5

    assert reaction_energy_difference(neutral, deprotonated, route="acid") > 0
    assert reaction_energy_difference(neutral, protonated, route="conjugate_acid") > 0
    assert reaction_energy_difference(neutral, protonated, route="pkb") < 0


def test_conjugate_acid_reaction_energy_has_positive_pkaH_orientation():
    results = pd.DataFrame(
        [
            {"molecule_id": "B", "level": "L", "state": "neutral", "state_id": "n", "site_id": None, "charge": 0, "gibbs_free_energy_eh": -100.0, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p_stable", "site_id": "N1", "charge": 1, "gibbs_free_energy_eh": -100.50, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p_high", "site_id": "N2", "charge": 1, "gibbs_free_energy_eh": -100.45, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )

    energies = reaction_energy_table(results, route="conjugate_acid")
    selected = energies[energies["selected"]].iloc[0]

    assert set(energies["property"]) == {"pKaH"}
    assert set(energies["energy_definition"]) == {"G(B) - G(BH+)"}
    assert selected["state_id"] == "p_stable"
    assert selected["reaction_energy_kj_mol"] == energies["reaction_energy_kj_mol"].max()
    assert selected["reaction_energy_kj_mol"] > 0


def test_conjugate_acid_prediction_is_labelled_pkaH():
    results = pd.DataFrame(
        [
            {"molecule_id": "B", "level": "L", "state": "neutral", "state_id": "n", "site_id": None, "gibbs_free_energy_eh": -100.0, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p", "site_id": "N", "gibbs_free_energy_eh": -100.5, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )
    refs = pd.DataFrame(
        [{"level": "L", "reference_kind": "proton", "value_kj_mol": -1100.0}]
    )

    pred = predict_acid_base(results, refs, route="conjugate_acid")

    assert set(pred["property"]) == {"pKaH"}


def test_conjugate_acid_selects_stable_protonated_state_and_max_pka():
    results = pd.DataFrame(
        [
            {"molecule_id": "B", "level": "L", "state": "neutral", "state_id": "n", "site_id": None, "gibbs_free_energy_eh": -100.0, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p_stable", "site_id": "N1", "gibbs_free_energy_eh": -100.50, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p_high", "site_id": "N2", "gibbs_free_energy_eh": -100.45, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )
    refs = pd.DataFrame([{"level": "L", "reference_kind": "proton", "value_kj_mol": -1100.0}])
    pred = predict_acid_base(results, refs, route="conjugate_acid")
    selected = pred[pred["selected"]]
    assert selected.iloc[0]["state_id"] == "p_stable"
    assert selected.iloc[0]["value"] == pred["value"].max()


def test_generic_pkah_table_selects_maximum_without_route_column():
    pred = pd.DataFrame(
        {
            "molecule_id": ["B", "B"],
            "level": ["L", "L"],
            "property": ["pKaH", "pKaH"],
            "state_id": ["p_N", "p_O"],
            "value": [9.5, -6.8],
        }
    )
    selected = select_macro_microstates(pred)
    assert selected.loc[selected["selected"], "state_id"].tolist() == ["p_N"]


def test_direct_pkb_selects_minimum():
    results = pd.DataFrame(
        [
            {"molecule_id": "B", "level": "L", "state": "neutral", "state_id": "n", "site_id": None, "gibbs_free_energy_eh": -100.0, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p1", "site_id": "N1", "gibbs_free_energy_eh": -100.50, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "B", "level": "L", "state": "protonated", "state_id": "p2", "site_id": "N2", "gibbs_free_energy_eh": -100.45, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )
    refs = pd.DataFrame([{"level": "L", "reference_kind": "hydroxide_minus_water", "value_kj_mol": 1300.0}])
    pred = predict_acid_base(results, refs, route="pkb")
    selected = pred[pred["selected"]]
    assert selected.iloc[0]["value"] == pred["value"].min()


def test_pkb_uses_neutral_to_protonated_and_pkb_from_pka_uses_conjugate_acid():
    results = pd.DataFrame(
        [
            {"molecule_id": "HA", "level": "L", "state": "neutral", "state_id": "n", "site_id": None, "charge": 0, "gibbs_free_energy_eh": -100.0, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "HA", "level": "L", "state": "deprotonated", "state_id": "a", "site_id": "Oacid", "charge": -1, "gibbs_free_energy_eh": -99.9, "method": "PBE0", "basis": "b", "valid": True},
            {"molecule_id": "HA", "level": "L", "state": "protonated", "state_id": "h2a", "site_id": "Ocarbonyl", "charge": 1, "gibbs_free_energy_eh": -100.5, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )
    proton_ref = pd.DataFrame(
        [{"level": "L", "reference_kind": "proton", "value_kj_mol": -1100.0}]
    )
    direct_ref = pd.DataFrame(
        [{"level": "L", "reference_kind": "hydroxide_minus_water", "value_kj_mol": 1300.0}]
    )

    direct = predict_acid_base(results, direct_ref, route="pkb")
    assert set(direct["state_id"]) == {"h2a"}

    pka_bh = predict_acid_base(results, proton_ref, route="conjugate_acid")
    via = predict_acid_base(results, proton_ref, route="pkb_from_pka", pkw=14.0)
    assert set(via["state_id"]) == {"h2a"}
    assert abs(via.loc[0, "value"] - (14.0 - pka_bh.loc[0, "value"])) < 1e-12


def test_state_semantics_rejects_relabelled_carboxylate_as_neutral():
    results = pd.DataFrame(
        [
            # This is the old notebook hack: A- relabelled as neutral.
            {"molecule_id": "HA", "level": "L", "state": "neutral", "state_id": "anion", "site_id": None, "charge": -1, "gibbs_free_energy_eh": -100.0, "method": "PBE0", "basis": "b", "valid": True},
            # HA relabelled as protonated.
            {"molecule_id": "HA", "level": "L", "state": "protonated", "state_id": "acid", "site_id": "O", "charge": 0, "gibbs_free_energy_eh": -100.5, "method": "PBE0", "basis": "b", "valid": True},
        ]
    )
    refs = pd.DataFrame(
        [{"level": "L", "reference_kind": "hydroxide_minus_water", "value_kj_mol": 1300.0}]
    )
    import pytest
    with pytest.raises(ValueError, match="neutral.*charge 0"):
        predict_acid_base(results, refs, route="pkb")


def test_explicit_hydroxide_minus_water_reference():
    from pka_calculator.thermodynamics.core import hydroxide_minus_water_reference_from_results

    results = pd.DataFrame(
        [
            {"molecule_id": "water_reference", "level": "L", "state": "custom", "state_id": "water", "charge": 0, "gibbs_free_energy_eh": -76.0, "valid": True},
            {"molecule_id": "hydroxide_reference", "level": "L", "state": "custom", "state_id": "hydroxide", "charge": -1, "gibbs_free_energy_eh": -75.5, "valid": True},
        ]
    )
    refs = hydroxide_minus_water_reference_from_results(results)
    expected = 0.5 * HARTREE_TO_KJ_MOL
    assert abs(refs.loc[0, "value_kj_mol"] - expected) < 1e-9
    assert refs.loc[0, "reference_kind"] == "hydroxide_minus_water"
