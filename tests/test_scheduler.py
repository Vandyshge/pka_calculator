from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

from pka_calculator.config import OrcaConfig, SlurmConfig
from pka_calculator.models import CalculationLevel, CalculationSpec, MoleculeState
from pka_calculator.schedulers.slurm import (
    _batch_script,
    pack_calculations,
    prepare_batch_scripts,
    query_slurm,
    submit_batches,
)


def make_spec(i: int, cores: int) -> CalculationSpec:
    state = MoleculeState(
        molecule_id=f"mol{i}",
        state_id="neutral",
        state="neutral",
        xyz=Path(__file__),
        charge=0,
        multiplicity=1,
    )
    level = CalculationLevel(name=f"l{cores}", method="PBE0", basis="def2-SVP", cores=cores)
    return CalculationSpec(f"c{i}", state, level, Path.cwd() / f"run{i}")


def test_pack_many_calculations_into_long_jobs():
    specs = [make_spec(i, 4) for i in range(5)]
    batches = pack_calculations(specs, node_cores=8)
    assert [len(batch) for batch in batches] == [5]

    batches = pack_calculations(specs, node_cores=8, calculations_per_job=2)
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_pack_separates_resource_classes():
    specs = [make_spec(1, 6), make_spec(2, 4), make_spec(3, 2), make_spec(4, 2)]
    batches = pack_calculations(specs, node_cores=8)
    assert len(batches) == 3
    assert all(len({spec.level.cores for spec in batch}) == 1 for batch in batches)
    assert [batch[0].level.cores for batch in batches] == [6, 4, 2]


def test_worker_pool_batch_script_uses_slurm_mpi_slots_not_srun(tmp_path):
    specs = [make_spec(i, 4) for i in range(5)]
    script = _batch_script(
        1,
        specs,
        tmp_path,
        OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=500),
    )

    # 8 physical cores / 4 cores per ORCA = 2 worker slots.
    assert "#SBATCH --ntasks=8" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "#SBATCH --hint=nomultithread" in script
    assert "#SBATCH --signal=B:TERM@60" in script
    assert "--cores-per-calc 4 --workers 2" in script
    assert "dispatcher.py" in script
    assert "queue.csv" in script
    assert "srun " not in script
    assert "mpirun " not in script


def test_max_parallel_calculations_reduces_allocation(tmp_path):
    specs = [make_spec(i, 4) for i in range(10)]
    script = _batch_script(
        1,
        specs,
        tmp_path,
        OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=32, max_parallel_calculations=3),
    )
    assert "#SBATCH --ntasks=12" in script
    assert "--cores-per-calc 4 --workers 3" in script


def test_prepare_batch_scripts_writes_standalone_dispatcher_and_queue(tmp_path):
    specs = [make_spec(i, 4) for i in range(5)]
    df = prepare_batch_scripts(
        specs,
        tmp_path,
        OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=3),
    )

    assert len(df) == 2
    first = tmp_path / "jobs" / "batch_0001"
    assert (first / "dispatcher.py").exists()
    assert (first / "queue.csv").exists()
    assert (first / "run.sh").exists()
    assert df.iloc[0]["worker_slots"] == 2
    assert df.iloc[0]["allocated_cores"] == 8


def _write_test_config(
    project_dir: Path, *, job_array: bool = True, array_max_size: int = 1000
) -> None:
    (project_dir / "config.toml").write_text(
        '[project]\nname = "array_test"\n'
        '[slurm]\nnode_cores = 8\ncalculations_per_job = 3\n'
        f'job_array = {str(job_array).lower()}\n'
        f'array_max_parallel = 2\narray_max_size = {array_max_size}\n'
        '[[levels]]\nname = "l4"\nmethod = "PBE0"\ncores = 4\n',
        encoding="utf-8",
    )


def test_submit_array_maps_tasks_and_resumes_without_duplicates(tmp_path, monkeypatch):
    specs = [make_spec(i, 4) for i in range(9)]
    prepare_batch_scripts(
        specs, tmp_path, OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=3),
    )
    _write_test_config(tmp_path)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr("pka_calculator.schedulers.slurm.subprocess.run", fake_run)
    first = submit_batches(tmp_path)
    assert len(commands) == 1
    assert commands[0][:3] == ["sbatch", "--parsable", "--array=0-2%2"]
    assert first.job_id.tolist() == ["12345", "", ""]
    assert first.array_task_job_id.tolist() == ["12345_0", "12345_1", "12345_2"]
    array_dir = tmp_path / "jobs" / "arrays" / "array_0001"
    script = (array_dir / "run.sh").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=8" in script
    assert 'exec bash "$batch_script"' in script
    assert len((array_dir / "task_scripts.txt").read_text().splitlines()) == 3
    assert (tmp_path / "jobs" / "batch_0002" / "submitted_job_id.txt").read_text() == "12345_1"

    second = submit_batches(tmp_path)
    assert len(commands) == 1
    assert second.job_id.tolist() == ["12345", "", ""]

    # The existing autonomous driver reads only nonempty job_id values to
    # construct afterany dependencies, so it now waits for the whole array.
    assert [x for x in second.job_id.tolist() if x] == ["12345"]


def test_array_wrapper_runs_the_selected_batch(tmp_path):
    from pka_calculator.schedulers.slurm import _array_script

    scripts = []
    for index in range(2):
        script = tmp_path / f"batch_{index}.sh"
        script.write_text(f"#!/bin/bash\necho {index}\n", encoding="utf-8")
        scripts.append(str(script))
    (tmp_path / "task_scripts.txt").write_text("\n".join(scripts) + "\n")
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(_array_script(tmp_path, 8, SlurmConfig(node_cores=8)))
    result = subprocess.run(
        ["bash", str(wrapper)],
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "1"},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1"


def test_arrays_split_at_configured_size(tmp_path, monkeypatch):
    specs = [make_spec(i, 4) for i in range(15)]
    prepare_batch_scripts(
        specs, tmp_path, OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=3),
    )
    _write_test_config(tmp_path, array_max_size=2)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=f"{700 + len(commands)}\n", stderr="")

    monkeypatch.setattr("pka_calculator.schedulers.slurm.subprocess.run", fake_run)
    submitted = submit_batches(tmp_path)
    assert len(commands) == 3
    assert submitted.job_id.tolist() == ["701", "", "702", "", "703"]
    assert submitted.array_task_job_id.tolist() == [
        "701_0", "701_1", "702_0", "702_1", "703_0",
    ]


def test_small_tail_batch_stays_in_one_array(tmp_path, monkeypatch):
    specs = [make_spec(i, 4) for i in range(4)]
    prepare_batch_scripts(
        specs, tmp_path, OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=3),
    )
    _write_test_config(tmp_path)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="80808\n", stderr="")

    monkeypatch.setattr("pka_calculator.schedulers.slurm.subprocess.run", fake_run)
    submitted = submit_batches(tmp_path)
    assert len(commands) == 1
    assert commands[0][2] == "--array=0-1%2"
    assert submitted.allocated_cores.tolist() == ["8", "4"]
    assert submitted.job_id.tolist() == ["80808", ""]
    assert "#SBATCH --ntasks=8" in (
        tmp_path / "jobs" / "arrays" / "array_0001" / "run.sh"
    ).read_text()


def test_query_slurm_expands_array_tasks_for_batch_status(tmp_path, monkeypatch):
    specs = [make_spec(i, 4) for i in range(6)]
    prepare_batch_scripts(
        specs, tmp_path, OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=3),
    )
    _write_test_config(tmp_path)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="45678\n", stderr="")
        assert command[:3] == ["squeue", "-r", "-h"]
        return SimpleNamespace(
            returncode=0,
            stdout="45678_0|RUNNING|00:01|node3\n45678_1|PENDING|0:00|\n",
            stderr="",
        )

    monkeypatch.setattr("pka_calculator.schedulers.slurm.subprocess.run", fake_run)
    submit_batches(tmp_path)
    status = query_slurm(tmp_path)
    assert status.state.tolist() == ["RUNNING", "PENDING"]
    assert status.queue_job_id.tolist() == ["45678_0", "45678_1"]


def test_legacy_individual_submission_can_be_selected(tmp_path, monkeypatch):
    specs = [make_spec(i, 4) for i in range(6)]
    prepare_batch_scripts(
        specs, tmp_path, OrcaConfig(executable="/opt/orca/orca"),
        SlurmConfig(node_cores=8, calculations_per_job=3),
    )
    _write_test_config(tmp_path, job_array=False)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=f"{100 + len(commands)}\n", stderr="")

    monkeypatch.setattr("pka_calculator.schedulers.slurm.subprocess.run", fake_run)
    submitted = submit_batches(tmp_path)
    assert len(commands) == 2
    assert all("--array=" not in " ".join(command) for command in commands)
    assert submitted.job_id.tolist() == ["101", "102"]
