from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .models import MoleculeState
from .utils import safe_name


REQUIRED_COLUMNS = {"molecule_id", "state_id", "state", "xyz", "charge", "multiplicity"}


def load_manifest(path: str | Path) -> list[MoleculeState]:
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    states: list[MoleculeState] = []
    seen: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        key = (str(row["molecule_id"]), str(row["state_id"]))
        if key in seen:
            raise ValueError(f"Duplicate molecule_id/state_id in manifest: {key}")
        seen.add(key)
        xyz = Path(str(row["xyz"]))
        if not xyz.is_absolute():
            xyz = (path.parent / xyz).resolve()
        if not xyz.exists():
            raise FileNotFoundError(f"XYZ not found for {key}: {xyz}")

        metadata = {}
        if "metadata" in df.columns and pd.notna(row.get("metadata")):
            try:
                metadata = json.loads(str(row["metadata"]))
            except json.JSONDecodeError:
                metadata = {"raw": str(row["metadata"])}

        states.append(
            MoleculeState(
                molecule_id=str(row["molecule_id"]),
                state_id=str(row["state_id"]),
                state=str(row["state"]).lower(),
                xyz=xyz,
                charge=int(row["charge"]),
                multiplicity=int(row["multiplicity"]),
                site_id=(str(row["site_id"]) if "site_id" in df.columns and pd.notna(row.get("site_id")) else None),
                parent_id=(str(row["parent_id"]) if "parent_id" in df.columns and pd.notna(row.get("parent_id")) else None),
                smiles=(str(row["smiles"]) if "smiles" in df.columns and pd.notna(row.get("smiles")) else None),
                metadata=metadata,
            )
        )
    return states


def write_manifest(states: list[MoleculeState], path: str | Path, *, relative_to: str | Path | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(relative_to).resolve() if relative_to else None
    rows = []
    for state in states:
        record = state.to_record()
        if base:
            try:
                record["xyz"] = str(Path(record["xyz"]).relative_to(base))
            except ValueError:
                pass
        record["metadata"] = json.dumps(record.get("metadata", {}), ensure_ascii=False)
        rows.append(record)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def infer_legacy_xyz_manifest(xyz_dir: str | Path, *, multiplicity: int = 1) -> list[MoleculeState]:
    """Import the old filename convention without pretending it encodes spin chemistry.

    Charge is inferred from suffix only. Multiplicity is deliberately supplied by the
    caller and defaults to singlet for migration convenience; users should review it.
    """
    xyz_dir = Path(xyz_dir).expanduser().resolve()
    states: list[MoleculeState] = []
    for xyz in sorted(xyz_dir.glob("*.xyz")):
        stem = xyz.stem
        state = "neutral"
        charge = 0
        base = stem
        site_id = None

        if stem.endswith("_deprotonated"):
            state, charge = "deprotonated", -1
            prefix = stem[: -len("_deprotonated")]
            match = re.match(r"^(.*)_([^_]+)$", prefix)
            if match:
                base, site_id = match.group(1), match.group(2)
            else:
                base = prefix
        elif stem.endswith("_protonated"):
            state, charge = "protonated", 1
            prefix = stem[: -len("_protonated")]
            match = re.match(r"^(.*)_([^_]+)$", prefix)
            if match:
                base, site_id = match.group(1), match.group(2)
            else:
                base = prefix

        state_id = safe_name(stem[len(base):].strip("_") or "neutral")
        states.append(
            MoleculeState(
                molecule_id=base,
                state_id=state_id,
                state=state,
                xyz=xyz,
                charge=charge,
                multiplicity=int(multiplicity),
                site_id=site_id,
                metadata={"imported_from_legacy_filename": True, "review_multiplicity": True},
            )
        )
    if not states:
        raise ValueError(f"No .xyz files found in {xyz_dir}")
    return states
