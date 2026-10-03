from pathlib import Path

from pka_calculator.engines.orca import OrcaParser, render_orca_input, validate_orca_result
from pka_calculator.models import CalculationLevel, CalculationSpec, MoleculeState


def spec(tmp_path: Path) -> CalculationSpec:
    xyz = tmp_path / "m.xyz"
    xyz.write_text("1\nH\nH 0 0 0\n")
    state = MoleculeState("m", "neutral", "neutral", xyz, 0, 1)
    level = CalculationLevel(
        name="test",
        method="PBE0",
        basis="def2-SVP",
        tasks=("OPT", "Freq"),
        cores=4,
        maxcore_mb=1000,
    )
    return CalculationSpec("calc", state, level, tmp_path)


def test_render_input(tmp_path):
    s = spec(tmp_path)
    text = render_orca_input(s)
    assert "! PBE0 def2-SVP TightSCF CPCM(water) OPT Freq" in text
    assert "nprocs 4" in text
    assert "%maxcore 1000" in text
    assert "* xyzfile 0 1 molecule.xyz" in text


def test_parse_and_validate_success(tmp_path):
    s = spec(tmp_path)
    output = tmp_path / "output.out"
    output.write_text(
        """
Program Version 6.1.1
THE OPTIMIZATION HAS CONVERGED
FINAL SINGLE POINT ENERGY      -100.500000000
Temperature                 ... 298.15 K
Zero point energy           ... 0.100000 Eh
Total Enthalpy              ... -100.390000 Eh
Final Gibbs free energy     ... -100.420000 Eh
   0:       0.00 cm**-1
   1:      -5.00 cm**-1
   2:     100.00 cm**-1
TOTAL RUN TIME: 0 days 0 hours 1 minutes 2 seconds 500 msec
ORCA TERMINATED NORMALLY
"""
    )
    result = OrcaParser().parse(s, output)
    validate_orca_result(result, s.level, imaginary_frequency_tolerance_cm1=-20.0)
    assert result.normal_termination
    assert result.optimization_converged
    assert result.gibbs_free_energy_eh == -100.42
    assert result.orca_runtime_s == 62.5
    assert result.orca_version == "6.1.1"
    assert result.valid


def test_imaginary_frequency_invalidates(tmp_path):
    s = spec(tmp_path)
    output = tmp_path / "output.out"
    output.write_text(
        """
THE OPTIMIZATION HAS CONVERGED
Final Gibbs free energy ... -100.0 Eh
   1: -120.0 cm**-1
ORCA TERMINATED NORMALLY
"""
    )
    result = OrcaParser().parse(s, output)
    validate_orca_result(result, s.level, imaginary_frequency_tolerance_cm1=-20.0)
    assert not result.valid
    assert "imaginary frequency" in result.validation_errors
