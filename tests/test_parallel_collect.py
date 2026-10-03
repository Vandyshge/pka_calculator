from pathlib import Path

import pandas as pd
import pytest

from pka_calculator.config import OrcaConfig, ProjectConfig, SlurmConfig, ThermodynamicsConfig
from pka_calculator.models import CalculationLevel, CalculationSpec, MoleculeState
from pka_calculator.workflow import collect_project_results


def test_parallel_collect_matches_serial_and_preserves_order(tmp_path, monkeypatch):
    level = CalculationLevel(name="xtb", method="XTB2", cores=1, tasks=("OPT",))
    specs = []
    for index in range(7):
        run_dir = tmp_path / f"run_{index}"
        run_dir.mkdir()
        (run_dir / "output.out").write_text(
            f"FINAL SINGLE POINT ENERGY     {-10.0 - index:.6f}\n"
            "THE OPTIMIZATION HAS CONVERGED\nORCA TERMINATED NORMALLY\n",
            encoding="utf-8",
        )
        (run_dir / "exit_code.txt").write_text("0", encoding="utf-8")
        state = MoleculeState(
            molecule_id=f"mol{index}", state_id="neutral", state="neutral",
            xyz=Path(__file__), charge=0, multiplicity=1,
        )
        specs.append(CalculationSpec(f"calc{index}", state, level, run_dir))

    config = ProjectConfig(
        project_name="parallel_test", orca=OrcaConfig(), slurm=SlurmConfig(),
        thermodynamics=ThermodynamicsConfig(), levels=(level,),
    )
    monkeypatch.setattr(
        "pka_calculator.workflow.load_project_specs", lambda _: (config, specs)
    )
    serial = collect_project_results(tmp_path, workers=1)
    parallel = collect_project_results(tmp_path, workers=3)
    pd.testing.assert_frame_equal(serial, parallel)
    assert parallel.calc_id.tolist() == [f"calc{i}" for i in range(7)]
    stored = pd.read_csv(tmp_path / "results.csv").fillna("")
    pd.testing.assert_frame_equal(parallel.fillna(""), stored, check_dtype=False)


def test_collect_rejects_nonpositive_workers(tmp_path):
    with pytest.raises(ValueError, match="workers must be >= 1"):
        collect_project_results(tmp_path, workers=0)
