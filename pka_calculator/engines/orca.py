from __future__ import annotations

import re
from pathlib import Path

from ..models import CalculationLevel, CalculationResult, CalculationSpec


TOTAL_TIME_PATTERN = re.compile(
    r"TOTAL RUN TIME:\s*"
    r"(?P<days>\d+)\s+days?\s+"
    r"(?P<hours>\d+)\s+hours?\s+"
    r"(?P<minutes>\d+)\s+minutes?\s+"
    r"(?P<seconds>\d+)\s+seconds?\s+"
    r"(?P<msec>\d+)\s+msec",
    flags=re.IGNORECASE,
)

FLOAT = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"
PATTERNS = {
    "final_single_point_energy_eh": [
        re.compile(rf"FINAL SINGLE POINT ENERGY\s+({FLOAT})", re.I),
    ],
    "gibbs_free_energy_eh": [
        re.compile(rf"Final Gibbs free energy\s+\.{{2,}}\s*({FLOAT})\s+Eh", re.I),
        re.compile(rf"Gibbs free energy\s+\.{{2,}}\s*({FLOAT})\s+Eh", re.I),
    ],
    "enthalpy_eh": [
        re.compile(rf"Total Enthalpy\s+\.{{2,}}\s*({FLOAT})\s+Eh", re.I),
        re.compile(rf"Final enthalpy\s+\.{{2,}}\s*({FLOAT})\s+Eh", re.I),
    ],
    "zero_point_energy_eh": [
        re.compile(rf"Zero point energy\s+\.{{2,}}\s*({FLOAT})\s+Eh", re.I),
        re.compile(rf"Zero point energy\s+({FLOAT})\s+Eh", re.I),
    ],
    "thermal_correction_gibbs_eh": [
        re.compile(rf"G-E\(el\)\s+\.{{2,}}\s*({FLOAT})\s+Eh", re.I),
        re.compile(rf"Thermal correction to Gibbs Free Energy\s+\.{{2,}}\s*({FLOAT})", re.I),
    ],
    "temperature_k": [
        re.compile(rf"Temperature\s+\.{{2,}}\s*({FLOAT})\s+K", re.I),
    ],
}

FREQUENCY_PATTERN = re.compile(rf"^\s*\d+\s*:\s*({FLOAT})\s+cm\*\*-1", re.I | re.M)
VERSION_PATTERN = re.compile(r"Program Version\s+([0-9][0-9A-Za-z_.-]*)", re.I)


def render_orca_input(spec: CalculationSpec, xyz_filename: str = "molecule.xyz") -> str:
    """Render ORCA input from a structured CalculationLevel."""
    level = spec.level
    keywords: list[str] = [level.method]
    if level.basis:
        keywords.append(level.basis)
    if level.dispersion:
        keywords.append(level.dispersion)
    keywords.extend(level.scf_keywords)
    if level.solvent_model and level.solvent:
        keywords.append(f"{level.solvent_model}({level.solvent})")
    keywords.extend(level.tasks)
    keywords.extend(level.extra_keywords)

    lines = ["! " + " ".join(str(x) for x in keywords), ""]
    if level.cores > 1:
        lines.extend(["%pal", f"  nprocs {level.cores}", "end", ""])
    if level.maxcore_mb:
        lines.extend([f"%maxcore {level.maxcore_mb}", ""])
    if level.wants_opt:
        lines.extend(["%geom", f"  MaxIter {level.geom_max_iter}", "end", ""])
    if level.extra_blocks.strip():
        lines.extend([level.extra_blocks.rstrip(), ""])
    state = spec.state
    lines.append(f"* xyzfile {state.charge} {state.multiplicity} {xyz_filename}")
    lines.append("")
    return "\n".join(lines)


def _last_float(text: str, patterns: list[re.Pattern[str]]) -> float | None:
    matches: list[re.Match[str]] = []
    for pattern in patterns:
        matches.extend(pattern.finditer(text))
    if not matches:
        return None
    matches.sort(key=lambda m: m.start())
    return float(matches[-1].group(1))


def parse_total_runtime(text: str) -> float | None:
    matches = list(TOTAL_TIME_PATTERN.finditer(text))
    if not matches:
        return None
    values = {k: int(v) for k, v in matches[-1].groupdict().items()}
    return (
        values["days"] * 86400
        + values["hours"] * 3600
        + values["minutes"] * 60
        + values["seconds"]
        + values["msec"] / 1000.0
    )


class OrcaParser:
    """Parse the subset of ORCA output needed for reproducible pKa/pKb workflows."""

    def parse(self, spec: CalculationSpec, output_file: str | Path) -> CalculationResult:
        output_file = Path(output_file)
        text = output_file.read_text(encoding="utf-8", errors="replace") if output_file.exists() else ""
        frequencies = [float(x) for x in FREQUENCY_PATTERN.findall(text)]
        version_matches = VERSION_PATTERN.findall(text)

        result = CalculationResult(
            calc_id=spec.calc_id,
            molecule_id=spec.state.molecule_id,
            state_id=spec.state.state_id,
            state=spec.state.state,
            site_id=spec.state.site_id,
            charge=spec.state.charge,
            multiplicity=spec.state.multiplicity,
            level=spec.level.name,
            method=spec.level.method,
            basis=spec.level.basis,
            cores=spec.level.cores,
            run_dir=str(spec.run_dir),
            normal_termination="ORCA TERMINATED NORMALLY" in text,
            optimization_converged=(
                ("THE OPTIMIZATION HAS CONVERGED" in text.upper())
                or ("HURRAY" in text.upper() and "OPTIMIZATION" in text.upper())
                if spec.level.wants_opt
                else None
            ),
            scf_converged=(
                False
                if re.search(r"SCF\s+(?:NOT|FAILED TO)\s+CONVERGE", text, re.I)
                else True if text else None
            ),
            final_single_point_energy_eh=_last_float(text, PATTERNS["final_single_point_energy_eh"]),
            gibbs_free_energy_eh=_last_float(text, PATTERNS["gibbs_free_energy_eh"]),
            enthalpy_eh=_last_float(text, PATTERNS["enthalpy_eh"]),
            zero_point_energy_eh=_last_float(text, PATTERNS["zero_point_energy_eh"]),
            thermal_correction_gibbs_eh=_last_float(text, PATTERNS["thermal_correction_gibbs_eh"]),
            temperature_k=_last_float(text, PATTERNS["temperature_k"]),
            imaginary_frequencies=(sum(x < 0 for x in frequencies) if frequencies else None),
            min_frequency_cm1=(min(frequencies) if frequencies else None),
            orca_runtime_s=parse_total_runtime(text),
            orca_version=(version_matches[-1] if version_matches else None),
        )
        return result


def validate_orca_result(
    result: CalculationResult,
    level: CalculationLevel,
    *,
    imaginary_frequency_tolerance_cm1: float = -20.0,
) -> CalculationResult:
    errors: list[str] = []
    if not result.normal_termination:
        errors.append("ORCA did not terminate normally")
    if result.scf_converged is False:
        errors.append("SCF convergence failure detected")
    if level.wants_opt and result.optimization_converged is not True:
        errors.append("geometry optimization convergence not confirmed")
    if level.wants_freq:
        if result.gibbs_free_energy_eh is None:
            errors.append("Gibbs free energy not found")
        if (
            result.min_frequency_cm1 is not None
            and result.min_frequency_cm1 < float(imaginary_frequency_tolerance_cm1)
        ):
            errors.append(
                f"imaginary frequency below tolerance: {result.min_frequency_cm1:.3f} cm^-1"
            )
    elif result.final_single_point_energy_eh is None:
        errors.append("final single point energy not found")
    result.validation_errors = "; ".join(errors)
    result.valid = not errors
    return result
