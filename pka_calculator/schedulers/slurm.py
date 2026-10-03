from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from ..config import OrcaConfig, ProjectConfig, SlurmConfig
from ..models import CalculationSpec
from ..utils import read_optional_text, safe_name


def pack_calculations(
    specs: list[CalculationSpec],
    *,
    node_cores: int,
    calculations_per_job: int | None = None,
) -> list[list[CalculationSpec]]:
    """Split calculations into homogeneous long-running worker-pool batches.

    Unlike the 1.0 scheduler, ``calculations_per_job`` limits the *total* number
    of calculations processed sequentially by one Slurm job. It does not limit
    concurrency. Concurrency is determined later from ``node_cores`` and the
    common per-calculation core count.

    Calculations with different ``level.cores`` values are deliberately placed
    in separate Slurm jobs. This keeps every worker slot in a job the same size
    and preserves the CPU-isolation strategy validated for ORCA/OpenMPI.
    """
    if node_cores < 1:
        raise ValueError("node_cores must be >= 1")
    if calculations_per_job is not None and calculations_per_job < 1:
        raise ValueError("calculations_per_job must be >= 1")

    by_cores: "OrderedDict[int, list[CalculationSpec]]" = OrderedDict()
    for spec in specs:
        cores = int(spec.level.cores)
        if cores < 1:
            raise ValueError(f"Calculation {spec.calc_id} requests invalid cores={cores}")
        if cores > node_cores:
            raise ValueError(
                f"Calculation {spec.calc_id} needs {cores} cores; node_cores={node_cores}"
            )
        by_cores.setdefault(cores, []).append(spec)

    batches: list[list[CalculationSpec]] = []
    for group in by_cores.values():
        chunk_size = calculations_per_job or len(group)
        for start in range(0, len(group), chunk_size):
            batches.append(group[start : start + chunk_size])
    return batches


def _worker_count(
    specs: list[CalculationSpec],
    slurm: SlurmConfig,
) -> tuple[int, int, int]:
    if not specs:
        raise ValueError("Cannot create a Slurm job for an empty batch")
    cores = int(specs[0].level.cores)
    if any(int(spec.level.cores) != cores for spec in specs):
        raise ValueError("Worker-pool Slurm batches must have one cores_per_calculation value")

    capacity = slurm.node_cores // cores
    if slurm.max_parallel_calculations is not None:
        capacity = min(capacity, slurm.max_parallel_calculations)
    workers = min(len(specs), capacity)
    if workers < 1:
        raise ValueError(
            f"No worker fits: cores_per_calculation={cores}, node_cores={slurm.node_cores}"
        )
    allocated_cores = workers * cores
    return cores, workers, allocated_cores


def _batch_script(
    batch_id: int,
    specs: list[CalculationSpec],
    batch_dir: Path,
    orca: OrcaConfig,
    slurm: SlurmConfig,
) -> str:
    cores_per_calc, workers, allocated_cores = _worker_count(specs, slurm)

    sbatch = [
        "#!/bin/bash",
        f"#SBATCH --job-name={safe_name(f'pka_{batch_id:04d}')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={allocated_cores}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --hint=nomultithread",
        f"#SBATCH --time={slurm.walltime}",
        f"#SBATCH --output={batch_dir}/slurm-%j.out",
        f"#SBATCH --error={batch_dir}/slurm-%j.err",
    ]
    if slurm.signal_seconds_before_end > 0:
        sbatch.append(f"#SBATCH --signal=B:TERM@{slurm.signal_seconds_before_end}")
    if slurm.partition:
        sbatch.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.account:
        sbatch.append(f"#SBATCH --account={slurm.account}")
    if slurm.nodelist:
        sbatch.append(f"#SBATCH --nodelist={slurm.nodelist}")
    if slurm.memory:
        sbatch.append(f"#SBATCH --mem={slurm.memory}")
    if slurm.exclusive:
        sbatch.append("#SBATCH --exclusive")
    sbatch.extend(str(line) for line in slurm.extra_sbatch_lines)

    module_lines = ["module purge"] if orca.modules else []
    module_lines.extend(f"module load {shlex.quote(module)}" for module in orca.modules)
    env_lines = [
        f"export {key}={shlex.quote(value)}" for key, value in orca.environment.items()
    ]

    dispatcher = shlex.quote(str(batch_dir / "dispatcher.py"))
    queue = shlex.quote(str(batch_dir / "queue.csv"))
    root = shlex.quote(str(batch_dir))
    orca_exe = shlex.quote(orca.executable)
    preflight_arg = "" if slurm.worker_preflight else " --skip-preflight"

    body = [
        "",
        "set -u",
        *module_lines,
        *env_lines,
        f"ORCA_EXE={orca_exe}",
        'PYTHON_EXE="${PYTHON_EXE:-python3}"',
        "",
        'echo "===== Slurm allocation ====="',
        'echo "host=$(hostname)"',
        'echo "job_id=${SLURM_JOB_ID:-}"',
        'echo "ntasks=${SLURM_NTASKS:-}"',
        'echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"',
        'echo "job_cpus_per_node=${SLURM_JOB_CPUS_PER_NODE:-}"',
        'echo "tasks_per_node=${SLURM_TASKS_PER_NODE:-}"',
        "echo",
        f'echo "queue_calculations={len(specs)}"',
        f'echo "cores_per_calculation={cores_per_calc}"',
        f'echo "worker_slots={workers}"',
        f'echo "allocated_physical_cores={allocated_cores}"',
        "",
        'dispatcher_pid=""',
        "forward_signal() {",
        '  if [[ -n "${dispatcher_pid}" ]] && kill -0 "${dispatcher_pid}" 2>/dev/null; then',
        '    echo "Forwarding termination signal to dispatcher ${dispatcher_pid}"',
        '    kill -TERM "${dispatcher_pid}" 2>/dev/null || true',
        "  fi",
        "}",
        "trap forward_signal TERM INT",
        "",
        f'"$PYTHON_EXE" {dispatcher} --root {root} --queue {queue} '
        f'--orca "$ORCA_EXE" --cores-per-calc {cores_per_calc} --workers {workers} '
        f'--poll {slurm.poll_interval_seconds:g}{preflight_arg} &',
        'dispatcher_pid="$!"',
        'wait "$dispatcher_pid"',
        'rc="$?"',
        '# A signal may interrupt bash wait before the dispatcher finishes cleanup.',
        'if kill -0 "$dispatcher_pid" 2>/dev/null; then',
        '  wait "$dispatcher_pid"',
        '  rc="$?"',
        'fi',
        'exit "$rc"',
        "",
    ]
    return "\n".join(sbatch + body)


def prepare_batch_scripts(
    specs: list[CalculationSpec],
    project_dir: str | Path,
    orca: OrcaConfig,
    slurm: SlurmConfig,
) -> pd.DataFrame:
    project_dir = Path(project_dir).expanduser().resolve()
    jobs_dir = project_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    batches = pack_calculations(
        specs,
        node_cores=slurm.node_cores,
        calculations_per_job=slurm.calculations_per_job,
    )

    worker_source = Path(__file__).with_name("worker_pool.py")
    records: list[dict[str, object]] = []

    for index, batch in enumerate(batches, start=1):
        batch_id = f"batch_{index:04d}"
        batch_dir = jobs_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        cores_per_calc, workers, allocated_cores = _worker_count(batch, slurm)

        dispatcher_path = batch_dir / "dispatcher.py"
        shutil.copy2(worker_source, dispatcher_path)
        dispatcher_path.chmod(0o755)

        queue_path = batch_dir / "queue.csv"
        with queue_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["calc_id", "workdir", "input_file"])
            for spec in batch:
                writer.writerow([spec.calc_id, str(spec.run_dir), "input.inp"])

        # Keep calculations.csv as a human-readable compatibility file.
        with (batch_dir / "calculations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["calc_id", "cores", "run_dir"])
            for spec in batch:
                writer.writerow([spec.calc_id, spec.level.cores, str(spec.run_dir)])

        script = batch_dir / "run.sh"
        script.write_text(
            _batch_script(index, batch, batch_dir, orca, slurm), encoding="utf-8"
        )
        script.chmod(0o755)

        records.append(
            {
                "batch_id": batch_id,
                "script_path": str(script),
                "queue_path": str(queue_path),
                "n_calculations": len(batch),
                "cores_per_calculation": cores_per_calc,
                "worker_slots": workers,
                "allocated_cores": allocated_cores,
                # Backwards-compatible display column used by 1.0 CLI output.
                "used_cores": allocated_cores,
                "job_id": "",
                "array_job_id": "",
                "array_task_id": "",
                "array_task_job_id": "",
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(project_dir / "batches.csv", index=False)
    return df


def _array_script(array_dir: Path, allocated_cores: int, slurm: SlurmConfig) -> str:
    """One Slurm allocation per array element; dispatch the indexed batch script."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=pka_array_{array_dir.name}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={allocated_cores}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --hint=nomultithread",
        f"#SBATCH --time={slurm.walltime}",
        f"#SBATCH --output={array_dir}/slurm-%A_%a.out",
        f"#SBATCH --error={array_dir}/slurm-%A_%a.err",
    ]
    if slurm.signal_seconds_before_end > 0:
        lines.append(f"#SBATCH --signal=B:TERM@{slurm.signal_seconds_before_end}")
    for key, value in (("partition", slurm.partition), ("account", slurm.account),
                       ("nodelist", slurm.nodelist), ("mem", slurm.memory)):
        if value:
            lines.append(f"#SBATCH --{key}={value}")
    if slurm.exclusive:
        lines.append("#SBATCH --exclusive")
    lines.extend(slurm.extra_sbatch_lines)
    manifest = shlex.quote(str(array_dir / "task_scripts.txt"))
    lines.extend([
        "",
        "set -euo pipefail",
        ': "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"',
        f"mapfile -t batch_scripts < {manifest}",
        'batch_script="${batch_scripts[SLURM_ARRAY_TASK_ID]:-}"',
        'if [[ -z "$batch_script" || ! -f "$batch_script" ]]; then',
        '  echo "No batch script for array task ${SLURM_ARRAY_TASK_ID}" >&2',
        "  exit 2",
        "fi",
        'exec bash "$batch_script"',
        "",
    ])
    return "\n".join(lines)


def _save_batches(df: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _recover_array_submissions(df: pd.DataFrame, project_dir: Path) -> None:
    """Recover task IDs after an interruption between sbatch and batches.csv."""
    index_by_batch = {str(row.batch_id): idx for idx, row in df.iterrows()}
    for receipt in sorted((project_dir / "jobs" / "arrays").glob("*/submission.json")):
        data = json.loads(receipt.read_text(encoding="utf-8"))
        base = str(data["array_job_id"])
        for task, batch_id in enumerate(data["batch_ids"]):
            idx = index_by_batch.get(batch_id)
            if idx is None or str(df.at[idx, "array_task_job_id"]):
                continue
            task_job_id = f"{base}_{task}"
            df.at[idx, "array_job_id"] = base
            df.at[idx, "array_task_id"] = str(task)
            df.at[idx, "array_task_job_id"] = task_job_id
            df.at[idx, "job_id"] = base if task == 0 else ""
            Path(df.at[idx, "script_path"]).parent.joinpath(
                "submitted_job_id.txt"
            ).write_text(task_job_id, encoding="utf-8")


def submit_batches(project_dir: str | Path, *, skip_submitted: bool = True) -> pd.DataFrame:
    project_dir = Path(project_dir).expanduser().resolve()
    path = project_dir / "batches.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run prepare first")
    df = pd.read_csv(path, dtype=str).fillna("")
    for column in ("array_job_id", "array_task_id", "array_task_job_id"):
        if column not in df:
            df[column] = ""
    _recover_array_submissions(df, project_dir)
    slurm = ProjectConfig.from_toml(project_dir / "config.toml").slurm

    pending: list[int] = []
    for idx, row in df.iterrows():
        id_file = Path(row["script_path"]).parent / "submitted_job_id.txt"
        existing = (read_optional_text(id_file) or str(row["array_task_job_id"])
                    or str(row["job_id"])).strip()
        if skip_submitted and existing:
            if not row["array_job_id"] and not row["job_id"]:
                df.at[idx, "job_id"] = existing
            continue
        pending.append(idx)

    if slurm.job_array:
        arrays_dir = project_dir / "jobs" / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        sequence = 1
        for offset in range(0, len(pending), slurm.array_max_size):
            chunk = pending[offset : offset + slurm.array_max_size]
            # Slurm requires one resource request for the whole array.
            # Using the maximum keeps even a small tail batch in this array.
            allocated_cores = max(int(df.at[idx, "allocated_cores"]) for idx in chunk)
            while (arrays_dir / f"array_{sequence:04d}").exists():
                sequence += 1
            array_dir = arrays_dir / f"array_{sequence:04d}"
            array_dir.mkdir()
            scripts = [str(df.at[idx, "script_path"]) for idx in chunk]
            if any("\n" in script or "\r" in script for script in scripts):
                raise ValueError("Batch script paths cannot contain newlines")
            (array_dir / "task_scripts.txt").write_text(
                "\n".join(scripts) + "\n", encoding="utf-8"
            )
            array_script = array_dir / "run.sh"
            array_script.write_text(
                _array_script(array_dir, allocated_cores, slurm), encoding="utf-8"
            )
            array_script.chmod(0o755)
            array_range = f"0-{len(chunk) - 1}"
            if slurm.array_max_parallel is not None:
                array_range += f"%{slurm.array_max_parallel}"
            result = subprocess.run(
                ["sbatch", "--parsable", f"--array={array_range}", str(array_script)],
                cwd=array_dir, text=True, capture_output=True, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"sbatch array failed: {(result.stderr or result.stdout).strip()}"
                )
            base = result.stdout.strip().split(";")[0]
            if not base.isdecimal():
                raise RuntimeError(f"Unrecognized sbatch array ID: {result.stdout!r}")
            (array_dir / "submission.json").write_text(json.dumps({
                "array_job_id": base,
                "batch_ids": [str(df.at[idx, "batch_id"]) for idx in chunk],
            }, indent=2), encoding="utf-8")
            for task, idx in enumerate(chunk):
                task_job_id = f"{base}_{task}"
                df.at[idx, "array_job_id"] = base
                df.at[idx, "array_task_id"] = str(task)
                df.at[idx, "array_task_job_id"] = task_job_id
                # One base ID per array lets external pipeline dependencies wait
                # for the whole array with afterany:<ArrayJobID>.
                df.at[idx, "job_id"] = base if task == 0 else ""
                Path(df.at[idx, "script_path"]).parent.joinpath(
                    "submitted_job_id.txt"
                ).write_text(task_job_id, encoding="utf-8")
            _save_batches(df, path)
            sequence += 1
    else:
        for idx in pending:
            row = df.loc[idx]
            batch_dir = Path(row["script_path"]).parent
            result = subprocess.run(
                ["sbatch", "--parsable", str(row["script_path"])],
                cwd=batch_dir, text=True, capture_output=True, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"sbatch failed for {row['batch_id']}: "
                    f"{(result.stderr or result.stdout).strip()}"
                )
            job_id = result.stdout.strip().split(";")[0]
            (batch_dir / "submitted_job_id.txt").write_text(job_id, encoding="utf-8")
            df.at[idx, "job_id"] = job_id
            _save_batches(df, path)
    _save_batches(df, path)
    return df


def _squeue(job_ids: list[str]) -> pd.DataFrame:
    if not job_ids:
        return pd.DataFrame(columns=["job_id", "state", "elapsed", "node"])
    result = subprocess.run(
        ["squeue", "-r", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%M|%N"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, state, elapsed, node = line.split("|", maxsplit=3)
        rows.append({"job_id": job_id, "state": state, "elapsed": elapsed, "node": node})
    return pd.DataFrame(rows)


def query_slurm(project_dir: str | Path) -> pd.DataFrame:
    project_dir = Path(project_dir).expanduser().resolve()
    batches = pd.read_csv(project_dir / "batches.csv", dtype=str).fillna("")
    if "array_task_job_id" not in batches:
        batches["array_task_job_id"] = ""
    batches["queue_job_id"] = batches["array_task_job_id"].where(
        batches["array_task_job_id"].ne(""), batches["job_id"]
    )
    ids = sorted({job_id.split("_")[0] for job_id in batches["queue_job_id"] if job_id})
    active = _squeue(ids)
    merged = batches.merge(active, left_on="queue_job_id", right_on="job_id", how="left",
                           suffixes=("", "_squeue"))
    merged["state"] = merged["state"].fillna("NOT_IN_QUEUE")
    return merged
