from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from .constants import DEFAULT_PKW_298, DEFAULT_TEMPERATURE_K
from .models import CalculationLevel


@dataclass
class OrcaConfig:
    executable: str = "orca"
    modules: tuple[str, ...] = ()
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )


@dataclass
class SlurmConfig:
    node_cores: int = 32
    calculations_per_job: int | None = None
    max_parallel_calculations: int | None = None
    job_array: bool = True
    array_max_parallel: int | None = None
    array_max_size: int = 1000
    walltime: str = "24:00:00"
    partition: str | None = None
    account: str | None = None
    nodelist: str | None = None
    memory: str | None = None
    exclusive: bool = False
    extra_sbatch_lines: tuple[str, ...] = ()
    poll_interval_seconds: float = 2.0
    worker_preflight: bool = True
    signal_seconds_before_end: int = 60

    def __post_init__(self) -> None:
        if self.node_cores < 1:
            raise ValueError("slurm.node_cores must be >= 1")
        if self.calculations_per_job is not None and self.calculations_per_job < 1:
            raise ValueError("slurm.calculations_per_job must be >= 1")
        if self.max_parallel_calculations is not None and self.max_parallel_calculations < 1:
            raise ValueError("slurm.max_parallel_calculations must be >= 1")
        if self.array_max_parallel is not None and self.array_max_parallel < 1:
            raise ValueError("slurm.array_max_parallel must be >= 1")
        if self.array_max_size < 1:
            raise ValueError("slurm.array_max_size must be >= 1")
        if self.poll_interval_seconds <= 0:
            raise ValueError("slurm.poll_interval_seconds must be > 0")
        if self.signal_seconds_before_end < 0:
            raise ValueError("slurm.signal_seconds_before_end must be >= 0")


@dataclass
class ThermodynamicsConfig:
    temperature_k: float = DEFAULT_TEMPERATURE_K
    pkw: float = DEFAULT_PKW_298
    imaginary_frequency_tolerance_cm1: float = -20.0


@dataclass
class ProjectConfig:
    project_name: str
    orca: OrcaConfig
    slurm: SlurmConfig
    thermodynamics: ThermodynamicsConfig
    levels: tuple[CalculationLevel, ...]

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        orca_raw = data.get("orca", {})
        slurm_raw = data.get("slurm", {})
        thermo_raw = data.get("thermodynamics", {})
        level_raw = data.get("levels", [])
        if not level_raw:
            raise ValueError("Configuration must contain at least one [[levels]] table")

        orca = OrcaConfig(
            executable=str(orca_raw.get("executable", "orca")),
            modules=tuple(str(x) for x in orca_raw.get("modules", [])),
            environment={str(k): str(v) for k, v in orca_raw.get("environment", {}).items()}
            or OrcaConfig().environment,
        )
        slurm = SlurmConfig(
            node_cores=int(slurm_raw.get("node_cores", 32)),
            calculations_per_job=(
                int(slurm_raw["calculations_per_job"])
                if slurm_raw.get("calculations_per_job") is not None
                else None
            ),
            max_parallel_calculations=(
                int(slurm_raw["max_parallel_calculations"])
                if slurm_raw.get("max_parallel_calculations") is not None
                else None
            ),
            job_array=bool(slurm_raw.get("job_array", True)),
            array_max_parallel=(
                int(slurm_raw["array_max_parallel"])
                if slurm_raw.get("array_max_parallel") is not None
                else None
            ),
            array_max_size=int(slurm_raw.get("array_max_size", 1000)),
            walltime=str(slurm_raw.get("walltime", "24:00:00")),
            partition=slurm_raw.get("partition"),
            account=slurm_raw.get("account"),
            nodelist=slurm_raw.get("nodelist"),
            memory=slurm_raw.get("memory"),
            exclusive=bool(slurm_raw.get("exclusive", False)),
            extra_sbatch_lines=tuple(str(x) for x in slurm_raw.get("extra_sbatch_lines", [])),
            poll_interval_seconds=float(slurm_raw.get("poll_interval_seconds", 2.0)),
            worker_preflight=bool(slurm_raw.get("worker_preflight", True)),
            signal_seconds_before_end=int(slurm_raw.get("signal_seconds_before_end", 60)),
        )
        thermo = ThermodynamicsConfig(
            temperature_k=float(thermo_raw.get("temperature_k", DEFAULT_TEMPERATURE_K)),
            pkw=float(thermo_raw.get("pkw", DEFAULT_PKW_298)),
            imaginary_frequency_tolerance_cm1=float(
                thermo_raw.get("imaginary_frequency_tolerance_cm1", -20.0)
            ),
        )

        levels = []
        for raw in level_raw:
            levels.append(
                CalculationLevel(
                    name=str(raw["name"]),
                    method=str(raw["method"]),
                    basis=(str(raw["basis"]) if raw.get("basis") else None),
                    dispersion=(str(raw["dispersion"]) if raw.get("dispersion") else None),
                    solvent_model=(
                        str(raw["solvent_model"]) if raw.get("solvent_model") else None
                    ),
                    solvent=(str(raw["solvent"]) if raw.get("solvent") else None),
                    tasks=tuple(str(x) for x in raw.get("tasks", ["OPT", "Freq"])),
                    scf_keywords=tuple(str(x) for x in raw.get("scf_keywords", ["TightSCF"])),
                    extra_keywords=tuple(str(x) for x in raw.get("extra_keywords", [])),
                    cores=int(raw.get("cores", 1)),
                    maxcore_mb=(int(raw["maxcore_mb"]) if raw.get("maxcore_mb") else None),
                    geom_max_iter=int(raw.get("geom_max_iter", 200)),
                    extra_blocks=str(raw.get("extra_blocks", "")),
                )
            )

        for level in levels:
            if level.cores > slurm.node_cores:
                raise ValueError(
                    f"Level {level.name!r} requests {level.cores} cores, "
                    f"but slurm.node_cores={slurm.node_cores}"
                )

        return cls(
            project_name=str(project.get("name", path.parent.name or "pka_project")),
            orca=orca,
            slurm=slurm,
            thermodynamics=thermo,
            levels=tuple(levels),
        )


DEFAULT_CONFIG_TEXT = '''[project]
name = "pka_project"

[orca]
# Parallel ORCA should be called by a full path on the cluster.
executable = "/path/to/orca"
modules = ["openmpi4/4.1.1", "orca/6.1.1"]

[orca.environment]
OMP_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"

[slurm]
# Physical-core budget for one worker-pool Slurm job.
node_cores = 32
# Total calculations handled sequentially by one Slurm job.
# 5000 calculations with this value produce about 10 jobs per resource class.
calculations_per_job = 500
# Optional cap on simultaneously active ORCA calculations.
# With 32 physical cores and 4 cores/calculation the natural maximum is 8.
# max_parallel_calculations = 8
# Submit worker-pool batches as a compact Slurm job array.
job_array = true
# Optional cap on simultaneously running array tasks.
# array_max_parallel = 2
# Maximum tasks per array submission (must not exceed the cluster MaxArraySize - 1).
array_max_size = 1000
walltime = "24:00:00"
# partition = "compute"
# account = "project"
# nodelist = "node1"
# memory = "64G"
exclusive = false
# Dispatcher checks for completed workers and refills free slots at this cadence.
poll_interval_seconds = 2.0
# Validate OpenMPI worker-level CPU isolation before starting ORCA.
worker_preflight = true
# Slurm sends TERM this many seconds before walltime; unfinished work is resumable.
signal_seconds_before_end = 60

[thermodynamics]
temperature_k = 298.15
pkw = 14.0
# Frequencies below this value mark a minimum as invalid by default.
imaginary_frequency_tolerance_cm1 = -20.0

[[levels]]
name = "PBE0_6-31+Gstar"
method = "PBE0"
basis = "6-31+G*"
solvent_model = "CPCM"
solvent = "water"
tasks = ["OPT", "Freq"]
scf_keywords = ["TightSCF"]
cores = 4
maxcore_mb = 2000
geom_max_iter = 200

# Add more [[levels]] blocks to run a method/basis matrix.
'''


def write_default_config(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return path
