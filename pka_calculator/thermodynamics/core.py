from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import HARTREE_TO_KJ_MOL, rtln10_kj_mol


ROUTES = {"acid", "conjugate_acid", "pkb"}

ROUTE_REACTIONS = {
    "acid": "HA -> A- + H+",
    "conjugate_acid": "BH+ -> B + H+",
    "pkb": "B + H2O -> BH+ + OH-",
    "pkb_from_pka": "pKb(B) = pKw - pKa(BH+)",
}

ROUTE_PROPERTIES = {
    "acid": "pKa",
    "conjugate_acid": "pKaH",
    "pkb": "pKb",
}

ROUTE_ENERGY_DEFINITIONS = {
    "acid": "G(A-) - G(HA)",
    "conjugate_acid": "G(B) - G(BH+)",
    "pkb": "G(BH+) - G(B)",
}


def reaction_energy_difference(
    neutral_energy,
    candidate_energy,
    *,
    route: str,
):
    """Return the molecular energy term in the direction of ``route``.

    Inputs may be scalars, NumPy arrays, or pandas Series and must use the
    same units. Reference-species terms are intentionally omitted:

    - ``acid``: ``G(A-) - G(HA)``;
    - ``conjugate_acid``: ``G(B) - G(BH+)``;
    - ``pkb``: ``G(BH+) - G(B)``.

    A pK fitted directly against this descriptor therefore has the physically
    expected positive orientation for every route. In particular,
    ``G(BH+) - G(B)`` is a direct-pKb descriptor, not a pKaH descriptor.
    """
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}")
    if route == "conjugate_acid":
        return neutral_energy - candidate_energy
    return candidate_energy - neutral_energy


def select_macro_microstates(
    predictions: pd.DataFrame,
    *,
    value_column: str = "value",
    selected_column: str = "selected",
) -> pd.DataFrame:
    """Mark the thermodynamically relevant microscopic state in each group.

    Acid dissociation and pKb select the minimum microscopic value.  The pKaH
    of a conjugate acid selects the maximum microscopic value because parallel
    protonation pathways add to the macroscopic association constant.  A
    ``route`` column is preferred; ``property='pKaH'`` is accepted for generic
    calibrated tables that do not carry route metadata.
    """
    required = {"molecule_id", value_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    out = predictions.copy()
    group_columns = ["molecule_id"]
    for column in ["level", "route", "property"]:
        if column in out.columns:
            group_columns.append(column)
    out[selected_column] = False

    for _, group in out.groupby(group_columns, dropna=False):
        values = pd.to_numeric(group[value_column], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue

        route = ""
        if "route" in group.columns and group["route"].notna().any():
            route = str(group.loc[group["route"].notna(), "route"].iloc[0]).lower()
        prop = ""
        if "property" in group.columns and group["property"].notna().any():
            prop = str(group.loc[group["property"].notna(), "property"].iloc[0]).lower()
        maximize = route == "conjugate_acid" or (not route and prop in {"pkah", "pka_h"})
        target = values[valid].max() if maximize else values[valid].min()
        chosen = valid & np.isclose(values, target, rtol=0, atol=1e-12)
        out.loc[group.index, selected_column] = chosen.to_numpy()
    return out


@dataclass(frozen=True)
class ThermodynamicReference:
    level: str
    reference_kind: str
    value_kj_mol: float
    temperature_k: float
    estimator: str = "mean"
    std_kj_mol: float | None = None
    n: int | None = None

    def to_record(self) -> dict:
        return asdict(self)


def _clean_results(results: pd.DataFrame, *, require_valid: bool = True) -> pd.DataFrame:
    df = results.copy()
    if "gibbs_free_energy_eh" not in df.columns:
        raise ValueError("results table must contain gibbs_free_energy_eh")
    if require_valid and "valid" in df.columns:
        valid = df["valid"]
        if valid.dtype == object:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[valid]
    df = df.dropna(subset=["gibbs_free_energy_eh", "molecule_id", "level", "state"])
    df["g_kj_mol"] = df["gibbs_free_energy_eh"].astype(float) * HARTREE_TO_KJ_MOL
    return df


def validate_route_states(results: pd.DataFrame, route: str) -> None:
    """Validate chemical-state semantics for an acid/base thermodynamic route.

    The library uses ``neutral`` literally: the reference molecule B/HA must have
    formal charge 0.  For a one-proton transformation, ``deprotonated`` must be
    -1 and ``protonated`` must be +1.  This deliberately catches silent state
    relabelling (for example treating A- as ``neutral`` just to reuse a pKb
    formula), which changes the chemical property being calculated.

    Validation is performed only when a ``charge`` column is available, so old
    result tables without charge metadata remain readable.
    """
    if route == "pkb_from_pka":
        route = "conjugate_acid"
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)} or 'pkb_from_pka'")
    if "charge" not in results.columns:
        return

    df = results.dropna(subset=["molecule_id", "level", "state", "charge"]).copy()
    if df.empty:
        return
    df["charge"] = pd.to_numeric(df["charge"], errors="raise").astype(int)

    bad_neutral = df[(df["state"] == "neutral") & (df["charge"] != 0)]
    if not bad_neutral.empty:
        sample = bad_neutral[["molecule_id", "level", "state_id", "charge"]].head(5)
        raise ValueError(
            "State semantics error: 'neutral' must have charge 0. "
            "Do not relabel an ionic state as neutral to change thermodynamic routes. "
            f"Examples: {sample.to_dict(orient='records')}"
        )

    target_state = "deprotonated" if route == "acid" else "protonated"
    expected_charge = -1 if target_state == "deprotonated" else 1
    bad_target = df[(df["state"] == target_state) & (df["charge"] != expected_charge)]
    if not bad_target.empty:
        sample = bad_target[["molecule_id", "level", "state_id", "charge"]].head(5)
        raise ValueError(
            f"State semantics error for route={route!r} ({ROUTE_REACTIONS[route]}): "
            f"'{target_state}' must have charge {expected_charge:+d} for a neutral parent. "
            f"Examples: {sample.to_dict(orient='records')}"
        )


def _neutral_minima(df: pd.DataFrame) -> pd.DataFrame:
    neutral = df[df["state"] == "neutral"].copy()
    if neutral.empty:
        return neutral
    idx = neutral.groupby(["molecule_id", "level"])["g_kj_mol"].idxmin()
    return neutral.loc[idx].copy()


def _candidate_table(results: pd.DataFrame, route: str, *, require_valid: bool = True) -> pd.DataFrame:
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}")
    validate_route_states(results, route)
    df = _clean_results(results, require_valid=require_valid)
    neutral = _neutral_minima(df)[
        ["molecule_id", "level", "state_id", "g_kj_mol"]
    ].rename(columns={"state_id": "neutral_state_id", "g_kj_mol": "g_neutral_kj_mol"})

    target_state = "deprotonated" if route == "acid" else "protonated"
    candidates = df[df["state"] == target_state].copy()
    candidates = candidates.merge(neutral, on=["molecule_id", "level"], how="inner")
    candidates = candidates.rename(columns={"g_kj_mol": "g_candidate_kj_mol"})
    keep = [
        "molecule_id",
        "level",
        "state_id",
        "site_id",
        "neutral_state_id",
        "g_neutral_kj_mol",
        "g_candidate_kj_mol",
        "method",
        "basis",
    ]
    return candidates[[x for x in keep if x in candidates.columns]].copy()


def reaction_energy_table(
    results: pd.DataFrame,
    *,
    route: str,
    require_valid: bool = True,
) -> pd.DataFrame:
    """Build route-oriented molecular reaction energies for every microstate.

    ``reaction_energy_kj_mol`` excludes the constant proton or ``OH-/H2O``
    reference contribution. Its sign always follows the reaction named in
    :data:`ROUTE_REACTIONS`. ``selected`` marks the correct macrostate: minimum
    for acid pKa/direct pKb and maximum for conjugate-acid pKaH.
    """
    candidates = _candidate_table(results, route, require_valid=require_valid)
    candidates["reaction_energy_kj_mol"] = reaction_energy_difference(
        candidates["g_neutral_kj_mol"],
        candidates["g_candidate_kj_mol"],
        route=route,
    )
    candidates["reaction_energy_eh"] = (
        candidates["reaction_energy_kj_mol"] / HARTREE_TO_KJ_MOL
    )
    candidates["route"] = route
    candidates["property"] = ROUTE_PROPERTIES[route]
    candidates["reaction"] = ROUTE_REACTIONS[route]
    candidates["energy_definition"] = ROUTE_ENERGY_DEFINITIONS[route]
    return select_macro_microstates(
        candidates, value_column="reaction_energy_kj_mol"
    )


def _read_experimental(
    path: str | Path, value_column: str | None = None, property_hint: str | None = None
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    aliases = {
        "molecule_id": ["molecule_id", "Molecule", "Base_Molecule", "name"],
        "pka": ["pka_exp", "pKa (exp)", "pKa", "pka"],
        "pkb": ["pkb_exp", "pKb (exp)", "pKb", "pkb"],
    }
    mol_col = next((x for x in aliases["molecule_id"] if x in df.columns), None)
    if mol_col is None:
        raise ValueError("Experimental file must contain molecule_id (or Molecule/name)")
    if value_column is None:
        hint = (property_hint or "").lower()
        if hint == "pkb":
            possible = aliases["pkb"] + aliases["pka"]
        else:
            possible = aliases["pka"] + aliases["pkb"]
        value_column = next((x for x in possible if x in df.columns), None)
    if value_column is None or value_column not in df.columns:
        raise ValueError("Could not identify experimental pKa/pKb column")
    return df[[mol_col, value_column]].rename(
        columns={mol_col: "molecule_id", value_column: "experimental"}
    ).dropna()


def calibrate_reference(
    results: pd.DataFrame,
    experimental: pd.DataFrame,
    *,
    route: str,
    temperature_k: float = 298.15,
    estimator: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate an effective proton or OH-/H2O reference on a training set only."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}")
    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be mean or median")
    candidates = reaction_energy_table(results, route=route)
    if candidates.empty:
        raise ValueError(f"No valid candidates for route={route}")

    selected = candidates[candidates["selected"]].merge(
        experimental, on="molecule_id", how="inner"
    )
    if selected.empty:
        raise ValueError("No overlap between calculated molecule_id and experimental data")

    factor = rtln10_kj_mol(temperature_k)
    selected["reference_estimate_kj_mol"] = (
        selected["experimental"] * factor - selected["reaction_energy_kj_mol"]
    )
    kind = "proton" if route in {"acid", "conjugate_acid"} else "hydroxide_minus_water"

    refs = []
    for level, group in selected.groupby("level"):
        values = group["reference_estimate_kj_mol"].astype(float)
        value = float(values.mean() if estimator == "mean" else values.median())
        refs.append(
            ThermodynamicReference(
                level=str(level),
                reference_kind=kind,
                value_kj_mol=value,
                temperature_k=float(temperature_k),
                estimator=estimator,
                std_kj_mol=float(values.std(ddof=1)) if len(values) > 1 else None,
                n=int(len(values)),
            ).to_record()
        )
    return pd.DataFrame(refs), selected


def predict_acid_base(
    results: pd.DataFrame,
    references: pd.DataFrame,
    *,
    route: str,
    temperature_k: float = 298.15,
    pkw: float = 14.0,
) -> pd.DataFrame:
    """Predict site-specific and thermodynamic pKa/pKb values.

    Routes are chemically explicit:

    - ``acid``: HA -> A- + H+
    - ``conjugate_acid``: BH+ -> B + H+
    - ``pkb``: B + H2O -> BH+ + OH-
    - ``pkb_from_pka``: pKb(B) = pKw - pKa(BH+)

    In particular, ``pkb_from_pka`` never uses pKa(B) itself.
    """
    if route == "pkb_from_pka":
        pka = predict_acid_base(
            results,
            references,
            route="conjugate_acid",
            temperature_k=temperature_k,
            pkw=pkw,
        )
        pka["route"] = "pkb_from_pka"
        pka["value"] = float(pkw) - pka["value"]
        pka["property"] = "pKb"
        return select_macro_microstates(pka)
    if route not in ROUTES:
        raise ValueError("route must be acid, conjugate_acid, pkb, or pkb_from_pka")

    candidates = reaction_energy_table(results, route=route)
    ref_kind = "proton" if route in {"acid", "conjugate_acid"} else "hydroxide_minus_water"
    refs = references[references["reference_kind"] == ref_kind][
        ["level", "value_kj_mol"]
    ].copy()
    candidates = candidates.merge(refs, on="level", how="inner")
    if candidates.empty:
        raise ValueError(f"No matching {ref_kind!r} references for calculated levels")
    factor = rtln10_kj_mol(temperature_k)
    candidates["value"] = (
        candidates["reaction_energy_kj_mol"] + candidates["value_kj_mol"]
    ) / factor
    return select_macro_microstates(candidates)


def hydroxide_minus_water_reference_from_results(
    results: pd.DataFrame,
    *,
    water_molecule_id: str = "water_reference",
    hydroxide_molecule_id: str = "hydroxide_reference",
    require_valid: bool = True,
    temperature_k: float = 298.15,
) -> pd.DataFrame:
    """Build the explicit ``G(OH-) - G(H2O)`` reference from QC results.

    The lowest valid Gibbs-energy row for each reference species and calculation
    level is used.  The returned table can be passed directly to
    ``predict_acid_base(..., route='pkb')``.
    """
    df = _clean_results(results, require_valid=require_valid)
    refs = df[df["molecule_id"].isin([water_molecule_id, hydroxide_molecule_id])].copy()
    if refs.empty:
        raise ValueError("No water/hydroxide reference calculations found")

    idx = refs.groupby(["molecule_id", "level"])["g_kj_mol"].idxmin()
    refs = refs.loc[idx, ["molecule_id", "level", "g_kj_mol"]]
    pivot = refs.pivot(index="level", columns="molecule_id", values="g_kj_mol")
    missing = [
        name
        for name in [water_molecule_id, hydroxide_molecule_id]
        if name not in pivot.columns
    ]
    if missing:
        raise ValueError(f"Missing valid reference species: {missing}")

    rows = []
    for level, row in pivot.iterrows():
        value = float(row[hydroxide_molecule_id] - row[water_molecule_id])
        rows.append(
            ThermodynamicReference(
                level=str(level),
                reference_kind="hydroxide_minus_water",
                value_kj_mol=value,
                temperature_k=float(temperature_k),
                estimator="explicit_qc",
                std_kj_mol=None,
                n=1,
            ).to_record()
        )
    return pd.DataFrame(rows)


def save_reference_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_experimental(
    path: str | Path, value_column: str | None = None, property_hint: str | None = None
) -> pd.DataFrame:
    return _read_experimental(path, value_column=value_column, property_hint=property_hint)
