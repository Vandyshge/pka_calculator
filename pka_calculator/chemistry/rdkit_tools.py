from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..manifest import write_manifest
from ..models import MoleculeState
from ..utils import safe_name


@dataclass(frozen=True)
class GeneratedMol:
    smiles: str
    site_id: str | None
    state: str
    charge: int


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "RDKit is required for chemistry generation. Install pka-calculator[chem] "
            "or install RDKit in the active environment."
        ) from exc
    return Chem, AllChem, rdMolStandardize


def smiles_to_3d(
    smiles: str,
    xyz_file: str | Path,
    *,
    random_seed: int = 42,
    optimize: bool = True,
    max_iters: int = 1000,
) -> Path:
    """Generate a reproducible 3D structure using ETKDGv3 and MMFF/UFF."""
    Chem, AllChem, _ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"RDKit could not generate a 3D conformer for {smiles}")

    if optimize:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))

    xyz_file = Path(xyz_file)
    xyz_file.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToXYZFile(mol, str(xyz_file))
    return xyz_file


def enumerate_tautomers(smiles: str, *, max_tautomers: int = 50) -> list[str]:
    """Return unique RDKit tautomers as canonical SMILES.

    No force-field energy is used as a thermodynamic tautomer ranking. The returned
    structures are candidates for a subsequent quantum-chemical screening step.
    """
    Chem, _, rdMolStandardize = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    enumerator = rdMolStandardize.TautomerEnumerator()
    params = enumerator.GetParams()
    params.maxTautomers = int(max_tautomers)
    values = {Chem.MolToSmiles(x, canonical=True) for x in enumerator.Enumerate(mol)}
    return sorted(values)


def _explicit_h_mol(smiles: str):
    Chem, _, _ = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.AddHs(mol)


def _deprotonated_candidates(smiles: str, elements: Iterable[str]) -> list[GeneratedMol]:
    Chem, _, _ = _rdkit()
    mol = _explicit_h_mol(smiles)
    allowed = set(elements)
    output: dict[str, GeneratedMol] = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            continue
        h_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        for hydrogen in h_neighbors:
            rw = Chem.RWMol(mol)
            parent_idx = atom.GetIdx()
            h_idx = hydrogen.GetIdx()
            # Removing an atom changes indices only after the removal, so record parent
            # information before modifying the graph.
            parent = rw.GetAtomWithIdx(parent_idx)
            parent.SetFormalCharge(parent.GetFormalCharge() - 1)
            rw.RemoveAtom(h_idx)
            candidate = rw.GetMol()
            try:
                Chem.SanitizeMol(candidate)
            except Exception:
                continue
            candidate = Chem.RemoveHs(candidate)
            can = Chem.MolToSmiles(candidate, canonical=True)
            output.setdefault(
                can,
                GeneratedMol(
                    can,
                    f"atom{parent_idx}_{atom.GetSymbol()}",
                    "deprotonated",
                    int(Chem.GetFormalCharge(candidate)),
                ),
            )
    return list(output.values())


def _protonated_candidates(smiles: str, elements: Iterable[str]) -> list[GeneratedMol]:
    Chem, _, _ = _rdkit()
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    allowed = set(elements)
    output: dict[str, GeneratedMol] = {}
    for atom in base.GetAtoms():
        if atom.GetSymbol() not in allowed or atom.GetFormalCharge() > 0:
            continue
        rw = Chem.RWMol(Chem.AddHs(base))
        parent = rw.GetAtomWithIdx(atom.GetIdx())
        parent.SetFormalCharge(parent.GetFormalCharge() + 1)
        h_idx = rw.AddAtom(Chem.Atom("H"))
        rw.AddBond(atom.GetIdx(), h_idx, Chem.BondType.SINGLE)
        candidate = rw.GetMol()
        try:
            Chem.SanitizeMol(candidate)
        except Exception:
            continue
        candidate = Chem.RemoveHs(candidate)
        can = Chem.MolToSmiles(candidate, canonical=True)
        output.setdefault(
            can,
            GeneratedMol(
                    can,
                    f"atom{atom.GetIdx()}_{atom.GetSymbol()}",
                    "protonated",
                    int(Chem.GetFormalCharge(candidate)),
                ),
        )
    return list(output.values())


def generate_acid_base_states(
    molecule_id: str,
    smiles: str,
    output_dir: str | Path,
    *,
    include_neutral: bool = True,
    deprotonate: bool = True,
    protonate: bool = True,
    deprotonation_elements: Iterable[str] = ("O", "N", "S", "P", "C"),
    protonation_elements: Iterable[str] = ("N", "O", "S", "P"),
    random_seed: int = 42,
    multiplicity: int = 1,
    state_prefix: str = "",
) -> list[MoleculeState]:
    """Generate graph-aware neutral/protonated/deprotonated candidates from SMILES.

    This is an enumeration layer, not a pKa predictor. RDKit valence/sanitization is
    used to reject impossible graph edits; quantum chemistry is still required to rank
    the surviving states.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Chem, _, _ = _rdkit()
    parent_mol = Chem.MolFromSmiles(smiles)
    if parent_mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    parent_charge = int(Chem.GetFormalCharge(parent_mol))
    if parent_charge != 0:
        raise ValueError(
            "generate_acid_base_states expects a neutral parent SMILES (formal charge 0). "
            f"Got formal charge {parent_charge:+d} for {smiles!r}. "
            "For ionic parents, build MoleculeState records explicitly."
        )

    values: list[GeneratedMol] = []
    if include_neutral:
        values.append(
            GeneratedMol(
                Chem.MolToSmiles(parent_mol, canonical=True),
                None,
                "neutral",
                parent_charge,
            )
        )
    if deprotonate:
        values.extend(_deprotonated_candidates(smiles, deprotonation_elements))
    if protonate:
        values.extend(_protonated_candidates(smiles, protonation_elements))

    counters: dict[str, int] = {}
    states: list[MoleculeState] = []
    for item in values:
        counters[item.state] = counters.get(item.state, 0) + 1
        idx = counters[item.state]
        raw_state_id = "neutral" if item.state == "neutral" else f"{item.state}_{idx:02d}"
        state_id = f"{state_prefix}{raw_state_id}"
        charge = int(item.charge)
        xyz = output_dir / f"{safe_name(molecule_id)}__{state_id}.xyz"
        smiles_to_3d(item.smiles, xyz, random_seed=random_seed + len(states))
        states.append(
            MoleculeState(
                molecule_id=str(molecule_id),
                state_id=state_id,
                state=item.state,
                xyz=xyz,
                charge=charge,
                multiplicity=int(multiplicity),
                site_id=item.site_id,
                parent_id=f"{state_prefix}neutral" if item.state != "neutral" else None,
                smiles=item.smiles,
                metadata={"generator": "rdkit_graph_enumeration"},
            )
        )
    return states


def smiles_csv_to_manifest(
    csv_file: str | Path,
    output_dir: str | Path,
    *,
    manifest_name: str = "molecules.csv",
    enumerate_states: bool = True,
    tautomers: bool = False,
    max_tautomers: int = 20,
) -> Path:
    """Generate an XYZ set and manifest from a CSV containing name,smiles."""
    csv_file = Path(csv_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    xyz_dir = output_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_file)
    if not {"name", "smiles"}.issubset(df.columns):
        raise ValueError("SMILES CSV must contain columns: name, smiles")

    states: list[MoleculeState] = []
    for _, row in df.iterrows():
        name, smiles = str(row["name"]), str(row["smiles"])
        parent_smiles = [smiles]
        if tautomers:
            parent_smiles = enumerate_tautomers(smiles, max_tautomers=max_tautomers)
        for taut_idx, taut_smiles in enumerate(parent_smiles):
            mol_id = name
            prefix = "" if len(parent_smiles) == 1 else f"taut{taut_idx:02d}_"
            if enumerate_states:
                states.extend(
                    generate_acid_base_states(
                        mol_id, taut_smiles, xyz_dir, state_prefix=prefix, random_seed=42 + taut_idx * 100
                    )
                )
            else:
                state_id = f"{prefix}neutral"
                xyz = xyz_dir / f"{safe_name(mol_id)}__{state_id}.xyz"
                smiles_to_3d(taut_smiles, xyz, random_seed=42 + taut_idx)
                states.append(
                    MoleculeState(
                        molecule_id=mol_id,
                        state_id=state_id,
                        state="neutral",
                        xyz=xyz,
                        charge=0,
                        multiplicity=1,
                        smiles=taut_smiles,
                    )
                )
    return write_manifest(states, output_dir / manifest_name, relative_to=output_dir)
