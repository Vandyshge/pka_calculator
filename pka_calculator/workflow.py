from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .engines.orca import OrcaParser, render_orca_input, validate_orca_result
from .manifest import load_manifest, write_manifest
from .models import CalculationSpec, MoleculeState
from .schedulers.slurm import prepare_batch_scripts
from .utils import read_optional_text, safe_name, sha256_file, write_json


def prepare_project(
    manifest_path: str | Path,
    config_path: str | Path,
    project_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[list[CalculationSpec], pd.DataFrame]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()
    config = ProjectConfig.from_toml(config_path)
    states = load_manifest(manifest_path)

    if project_dir.exists() and any(project_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Project directory is not empty: {project_dir}. Use --overwrite to rebuild inputs."
        )
    project_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for child in ["calculations", "jobs", "states"]:
            shutil.rmtree(project_dir / child, ignore_errors=True)

    shutil.copy2(config_path, project_dir / "config.toml")
    states_dir = project_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    local_states: list[MoleculeState] = []
    for state in states:
        filename = f"{safe_name(state.molecule_id)}__{safe_name(state.state_id)}.xyz"
        target = states_dir / filename
        shutil.copy2(state.xyz, target)
        local_states.append(
            MoleculeState(
                molecule_id=state.molecule_id,
                state_id=state.state_id,
                state=state.state,
                xyz=target,
                charge=state.charge,
                multiplicity=state.multiplicity,
                site_id=state.site_id,
                parent_id=state.parent_id,
                smiles=state.smiles,
                metadata=state.metadata,
            )
        )
    write_manifest(local_states, project_dir / "molecules.csv", relative_to=project_dir)

    specs: list[CalculationSpec] = []
    rows = []
    for level in config.levels:
        for state in local_states:
            calc_id = safe_name(f"{level.name}__{state.molecule_id}__{state.state_id}")
            run_dir = (
                project_dir
                / "calculations"
                / safe_name(level.name)
                / safe_name(state.molecule_id)
                / safe_name(state.state_id)
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(state.xyz, run_dir / "molecule.xyz")
            spec = CalculationSpec(calc_id=calc_id, state=state, level=level, run_dir=run_dir)
            input_path = run_dir / "input.inp"
            input_path.write_text(render_orca_input(spec), encoding="utf-8")
            metadata = {
                "calc_id": calc_id,
                "state": state.to_record(),
                "level": level.to_record(),
                "input_sha256": sha256_file(input_path),
                "xyz_sha256": sha256_file(run_dir / "molecule.xyz"),
            }
            write_json(run_dir / "metadata.json", metadata)
            specs.append(spec)
            rows.append(
                {
                    "calc_id": calc_id,
                    "molecule_id": state.molecule_id,
                    "state_id": state.state_id,
                    "state": state.state,
                    "site_id": state.site_id,
                    "charge": state.charge,
                    "multiplicity": state.multiplicity,
                    "level": level.name,
                    "method": level.method,
                    "basis": level.basis,
                    "cores": level.cores,
                    "run_dir": str(run_dir),
                }
            )
    pd.DataFrame(rows).to_csv(project_dir / "calculations.csv", index=False)
    batches = prepare_batch_scripts(specs, project_dir, config.orca, config.slurm)
    return specs, batches


def load_project_specs(project_dir: str | Path) -> tuple[ProjectConfig, list[CalculationSpec]]:
    project_dir = Path(project_dir).expanduser().resolve()
    config = ProjectConfig.from_toml(project_dir / "config.toml")
    states = load_manifest(project_dir / "molecules.csv")
    state_map = {(s.molecule_id, s.state_id): s for s in states}
    level_map = {x.name: x for x in config.levels}
    calcs = pd.read_csv(project_dir / "calculations.csv")
    specs = []
    for _, row in calcs.iterrows():
        state = state_map[(str(row["molecule_id"]), str(row["state_id"]))]
        level = level_map[str(row["level"])]
        specs.append(
            CalculationSpec(
                calc_id=str(row["calc_id"]),
                state=state,
                level=level,
                run_dir=Path(str(row["run_dir"])),
            )
        )
    return config, specs



def rebuild_project_jobs(project_dir: str | Path) -> pd.DataFrame:
    """Rebuild only Slurm worker-pool jobs for an existing prepared project.

    Calculation directories and existing ORCA outputs are preserved. This is
    intended for scheduler upgrades or Slurm-configuration changes.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    config, specs = load_project_specs(project_dir)
    shutil.rmtree(project_dir / "jobs", ignore_errors=True)
    batches_path = project_dir / "batches.csv"
    if batches_path.exists():
        batches_path.unlink()
    return prepare_batch_scripts(specs, project_dir, config.orca, config.slurm)

def _collect_one(item: tuple[CalculationSpec, float]) -> dict[str, object]:
    spec, imaginary_frequency_tolerance_cm1 = item
    result = OrcaParser().parse(spec, spec.run_dir / "output.out")
    wall = read_optional_text(spec.run_dir / "wall_seconds.txt")
    exit_code = read_optional_text(spec.run_dir / "exit_code.txt")
    result.shell_runtime_s = float(wall) if wall else None
    result.exit_code = int(exit_code) if exit_code is not None else None
    result.node = read_optional_text(spec.run_dir / "node.txt")
    validate_orca_result(
        result,
        spec.level,
        imaginary_frequency_tolerance_cm1=imaginary_frequency_tolerance_cm1,
    )
    return result.to_record()


def collect_project_results(project_dir: str | Path, *, workers: int = 1) -> pd.DataFrame:
    """Parse outputs in calculation order; workers > 1 uses separate processes."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    project_dir = Path(project_dir).expanduser().resolve()
    config, specs = load_project_specs(project_dir)
    tolerance = config.thermodynamics.imaginary_frequency_tolerance_cm1
    items = ((spec, tolerance) for spec in specs)
    rows = []
    if workers == 1:
        parsed = map(_collect_one, items)
        for row in parsed:
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(_collect_one, items, chunksize=32):
                rows.append(row)
                if len(rows) % 5000 == 0:
                    print(f"Parsed {len(rows)}/{len(specs)} ORCA outputs", flush=True)
    df = pd.DataFrame(rows)
    temporary = project_dir / "results.csv.tmp"
    df.to_csv(temporary, index=False)
    temporary.replace(project_dir / "results.csv")
    return df


def project_progress(project_dir: str | Path) -> pd.DataFrame:
    project_dir = Path(project_dir).expanduser().resolve()
    calcs = pd.read_csv(project_dir / "calculations.csv")
    rows = []
    for _, row in calcs.iterrows():
        run_dir = Path(str(row["run_dir"]))
        output = run_dir / "output.out"
        text = output.read_text(encoding="utf-8", errors="replace") if output.exists() else ""
        if "ORCA TERMINATED NORMALLY" in text:
            status = "completed"
        elif (run_dir / "exit_code.txt").exists():
            status = "failed"
        elif output.exists():
            status = "running_or_interrupted"
        else:
            status = "not_started"
        rows.append({**row.to_dict(), "status": status})
    return pd.DataFrame(rows)
