from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


ALLOWED_STATES = {"neutral", "deprotonated", "protonated", "custom"}


@dataclass(frozen=True)
class MoleculeState:
    """One chemically explicit state used in a quantum-chemical calculation."""

    molecule_id: str
    state_id: str
    state: str
    xyz: Path
    charge: int
    multiplicity: int
    site_id: str | None = None
    parent_id: str | None = None
    smiles: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in ALLOWED_STATES:
            raise ValueError(f"Unknown state {self.state!r}; expected one of {sorted(ALLOWED_STATES)}")
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be >= 1")
        if not str(self.molecule_id).strip() or not str(self.state_id).strip():
            raise ValueError("molecule_id and state_id must be non-empty")
        object.__setattr__(self, "xyz", Path(self.xyz).expanduser().resolve())

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["xyz"] = str(self.xyz)
        return record


@dataclass(frozen=True)
class CalculationLevel:
    """A complete ORCA level of theory and resource request."""

    name: str
    method: str
    basis: str | None = None
    dispersion: str | None = None
    solvent_model: str | None = "CPCM"
    solvent: str | None = "water"
    tasks: Sequence[str] = ("OPT", "Freq")
    scf_keywords: Sequence[str] = ("TightSCF",)
    extra_keywords: Sequence[str] = ()
    cores: int = 1
    maxcore_mb: int | None = None
    geom_max_iter: int = 200
    extra_blocks: str = ""

    def __post_init__(self) -> None:
        if self.cores < 1:
            raise ValueError("cores must be >= 1")
        if self.maxcore_mb is not None and self.maxcore_mb < 1:
            raise ValueError("maxcore_mb must be >= 1")
        if not self.name.strip() or not self.method.strip():
            raise ValueError("level name and method must be non-empty")

    @property
    def wants_opt(self) -> bool:
        return any(str(x).lower() == "opt" for x in self.tasks)

    @property
    def wants_freq(self) -> bool:
        return any(str(x).lower() in {"freq", "numfreq"} for x in self.tasks)

    def to_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["tasks"] = list(self.tasks)
        result["scf_keywords"] = list(self.scf_keywords)
        result["extra_keywords"] = list(self.extra_keywords)
        return result


@dataclass(frozen=True)
class CalculationSpec:
    calc_id: str
    state: MoleculeState
    level: CalculationLevel
    run_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir).expanduser().resolve())


@dataclass
class CalculationResult:
    calc_id: str
    molecule_id: str
    state_id: str
    state: str
    site_id: str | None
    charge: int
    multiplicity: int
    level: str
    method: str
    basis: str | None
    cores: int
    run_dir: str
    normal_termination: bool = False
    optimization_converged: bool | None = None
    scf_converged: bool | None = None
    final_single_point_energy_eh: float | None = None
    gibbs_free_energy_eh: float | None = None
    enthalpy_eh: float | None = None
    zero_point_energy_eh: float | None = None
    thermal_correction_gibbs_eh: float | None = None
    temperature_k: float | None = None
    imaginary_frequencies: int | None = None
    min_frequency_cm1: float | None = None
    orca_runtime_s: float | None = None
    shell_runtime_s: float | None = None
    exit_code: int | None = None
    orca_version: str | None = None
    node: str | None = None
    valid: bool = False
    validation_errors: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
