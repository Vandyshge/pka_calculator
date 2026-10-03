from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .constants import HARTREE_TO_KCAL_MOL
from .utils import safe_name


def select_conformers_within_energy_window(
    results: pd.DataFrame,
    *,
    energy_window_kcal_mol: float = 3.0,
    max_conformers_per_molecule: int | None = 4,
    energy_column: str = "final_single_point_energy_eh",
    valid_column: str = "valid",
) -> pd.DataFrame:
    """Select low-energy conformers while preserving their parent identity.

    Selection is performed independently for every ``molecule_id`` and, when
    present, every calculation ``level``. All valid conformers within
    ``energy_window_kcal_mol`` of the lowest energy are retained, ordered by
    energy, and optionally capped by ``max_conformers_per_molecule``.

    The input energy must be in Hartree. The returned table contains
    ``relative_energy_kcal_mol``, ``conformer_energy_rank``, and the selection
    parameters so downstream ion calculations can remain auditable.
    """
    if energy_window_kcal_mol < 0:
        raise ValueError("energy_window_kcal_mol must be >= 0")
    if max_conformers_per_molecule is not None and max_conformers_per_molecule < 1:
        raise ValueError("max_conformers_per_molecule must be >= 1 or None")

    required = {"molecule_id", energy_column}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")

    eligible = results.copy()
    if valid_column in eligible.columns:
        valid = eligible[valid_column]
        if valid.dtype == object:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        eligible = eligible[valid.fillna(False)].copy()
    eligible[energy_column] = pd.to_numeric(eligible[energy_column], errors="coerce")
    eligible = eligible.dropna(subset=[energy_column])
    if eligible.empty:
        return eligible.assign(
            relative_energy_kcal_mol=pd.Series(dtype=float),
            conformer_energy_rank=pd.Series(dtype="Int64"),
            energy_window_kcal_mol=pd.Series(dtype=float),
            max_conformers_per_molecule=pd.Series(dtype="Int64"),
        )

    group_columns = ["molecule_id"]
    if "level" in eligible.columns:
        group_columns.append("level")
    minima = eligible.groupby(group_columns)[energy_column].transform("min")
    eligible["relative_energy_kcal_mol"] = (
        eligible[energy_column] - minima
    ) * HARTREE_TO_KCAL_MOL
    selected = eligible[
        eligible["relative_energy_kcal_mol"] <= energy_window_kcal_mol + 1e-12
    ].copy()

    tie_breakers = [column for column in ("state_id", "calc_id") if column in selected.columns]
    selected = selected.sort_values(
        group_columns + ["relative_energy_kcal_mol"] + tie_breakers,
        kind="stable",
    )
    selected["conformer_energy_rank"] = (
        selected.groupby(group_columns).cumcount() + 1
    )
    if max_conformers_per_molecule is not None:
        selected = selected[
            selected["conformer_energy_rank"] <= max_conformers_per_molecule
        ].copy()
    selected["energy_window_kcal_mol"] = float(energy_window_kcal_mol)
    selected["max_conformers_per_molecule"] = max_conformers_per_molecule
    return selected.reset_index(drop=True)


def read_xyz_frames(path: str | Path) -> list[tuple[str, list[str]]]:
    """Read a concatenated XYZ trajectory without regex ambiguity."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom count at line {i + 1}: {lines[i]!r}") from exc
        if i + 1 + n >= len(lines) + 1:
            raise ValueError("Truncated XYZ frame")
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        atom_lines = lines[i + 2 : i + 2 + n]
        if len(atom_lines) != n:
            raise ValueError("Truncated XYZ frame")
        frames.append((comment, atom_lines))
        i += n + 2
    return frames


def write_xyz_frame(path: str | Path, comment: str, atom_lines: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"{len(atom_lines)}\n{comment}\n" + "\n".join(atom_lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def extract_final_geometries(
    project_dir: str | Path,
    output_dir: str | Path,
    *,
    level: str | None = None,
    valid_only: bool = True,
) -> pd.DataFrame:
    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calcs = pd.read_csv(project_dir / "calculations.csv")
    results_path = project_dir / "results.csv"
    if valid_only and results_path.exists():
        results = pd.read_csv(results_path)[["calc_id", "valid"]]
        calcs = calcs.merge(results, on="calc_id", how="left")
        valid = calcs["valid"]
        if valid.dtype == object:
            valid = valid.astype(str).str.lower().eq("true")
        calcs = calcs[valid.fillna(False)]
    if level:
        calcs = calcs[calcs["level"] == level]

    rows = []
    for _, row in calcs.iterrows():
        run_dir = Path(str(row["run_dir"]))
        trj = run_dir / "input_trj.xyz"
        target = output_dir / (
            f"{safe_name(str(row['molecule_id']))}__{safe_name(str(row['state_id']))}__"
            f"{safe_name(str(row['level']))}.xyz"
        )
        source = ""
        if trj.exists():
            frames = read_xyz_frames(trj)
            if frames:
                write_xyz_frame(target, frames[-1][0], frames[-1][1])
                source = str(trj)
        elif (run_dir / "molecule.xyz").exists():
            shutil.copy2(run_dir / "molecule.xyz", target)
            source = str(run_dir / "molecule.xyz")
        if source:
            rows.append({"calc_id": row["calc_id"], "output_xyz": str(target), "source": source})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "geometries.csv", index=False)
    return df
