import pytest

rdkit = pytest.importorskip("rdkit")

from pka_calculator.chemistry.rdkit_tools import generate_acid_base_states


def test_acetic_acid_protonation_and_deprotonation_charges(tmp_path):
    states = generate_acid_base_states(
        "acetic",
        "CC(=O)O",
        tmp_path,
        include_neutral=True,
        deprotonate=True,
        protonate=True,
        deprotonation_elements=("O",),
        protonation_elements=("O",),
    )
    by_state = {}
    for state in states:
        by_state.setdefault(state.state, []).append(state)

    assert {x.charge for x in by_state["neutral"]} == {0}
    assert {x.charge for x in by_state["deprotonated"]} == {-1}
    assert {x.charge for x in by_state["protonated"]} == {1}
    assert len(by_state["protonated"]) >= 2


def test_ionic_parent_rejected_even_when_neutral_state_not_requested(tmp_path):
    with pytest.raises(ValueError, match="neutral parent SMILES"):
        generate_acid_base_states(
            "acetate",
            "CC(=O)[O-]",
            tmp_path,
            include_neutral=False,
            deprotonate=False,
            protonate=True,
            protonation_elements=("O",),
        )
