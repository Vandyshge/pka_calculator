# pka-calculator 1.3.1

A reproducible Python workflow for quantum-chemical **pKa / pKb** calculations with
ORCA and Slurm.

## Installation

```bash
git clone https://github.com/Vandyshge/pka_calculator.git
cd pka_calculator
pip install -e .
```

To update an existing checkout on either cluster, activate its Python environment and run:

```bash
cd pka_calculator
git pull --ff-only
python -m pip install -e .
```

If the checkout contains local changes, save them before pulling; Git will stop rather than
overwrite them.

For structure generation from SMILES:

```bash
pip install -e '.[chem]'
```

For Plotly HTML reports:

```bash
pip install -e '.[interactive]'
```

Development/tests:

```bash
pip install -e '.[dev]'
pytest
```

Python 3.10+ is required. ORCA and Slurm are external programs and are not installed by
this package.

## 1. Initialize a project

```bash
pka-calculator init my_pka_project
cd my_pka_project
```

This creates:

```text
my_pka_project/
├── pka.toml
└── molecules.csv
```

Edit `pka.toml` for your cluster and calculation levels.

Example:

```toml
[orca]
executable = "/opt/calculation/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
modules = ["gnu9/9.4.0", "openmpi4/4.1.1", "orca/6.1.1"]

[slurm]
node_cores = 32
calculations_per_job = 500
# max_parallel_calculations = 8
job_array = true
# array_max_parallel = 2
array_max_size = 1000
walltime = "24:00:00"
exclusive = false
poll_interval_seconds = 2.0
worker_preflight = true
signal_seconds_before_end = 60

[[levels]]
name = "PBE0_6-31+Gstar"
method = "PBE0"
basis = "6-31+G*"
solvent_model = "CPCM"
solvent = "water"
tasks = ["OPT", "Freq"]
cores = 4
maxcore_mb = 2000
```

With `node_cores = 32` and `cores = 4`, one Slurm array element can keep up to eight ORCA
workers active. `calculations_per_job = 500` means that those workers dynamically consume a
queue of up to 500 calculations before the array element exits. For 5000 calculations this
produces roughly 10 array elements within one `sbatch` submission.
`array_max_size` must be below the cluster's `MaxArraySize`; larger projects are split into
the minimum number of arrays needed to respect that limit.
Slurm normally combines pending array elements in `squeue`; running elements can still appear
on separate lines. `squeue -r` explicitly expands all elements.

## 2. Molecule manifest

`molecules.csv` is the source of truth:

```csv
molecule_id,state_id,state,xyz,charge,multiplicity,site_id,parent_id
1,neutral,neutral,xyz/1.xyz,0,1,,
1,deprot_O1,deprotonated,xyz/1_deprot_O1.xyz,-1,1,O1,neutral
1,prot_N1,protonated,xyz/1_prot_N1.xyz,1,1,N1,neutral
```

Allowed standard states are:

- `neutral`
- `deprotonated`
- `protonated`
- `custom`

`multiplicity` is explicit because electron parity only distinguishes odd/even electron
counts; it cannot determine the correct spin state for general molecules or transition-metal
complexes.

### Migration from old XYZ filenames

```bash
pka-calculator import-xyz old_xyz/ -o molecules.csv --multiplicity 1
```

This only migrates the old naming convention. Review charges and especially multiplicities
before production calculations.

## 3. Generate structures from SMILES

Input CSV:

```csv
name,smiles
formic,O=CO
pyridine,n1ccccc1
```

Generate neutral, deprotonated and protonated candidate states:

```bash
pka-calculator generate-smiles molecules_smiles.csv generated/
```

Enumerate tautomers first:

```bash
pka-calculator generate-smiles molecules_smiles.csv generated/ \
  --tautomers --max-tautomers 20
```

The generator uses RDKit graph edits and sanitization to enumerate candidate sites. It is a
structure enumeration stage, not an acidity/basicity predictor.

## 4. Prepare ORCA calculations

```bash
pka-calculator prepare molecules.csv -c pka.toml -o run_001
```

The prepared project is self-contained:

```text
run_001/
├── config.toml
├── molecules.csv
├── states/
├── calculations.csv
├── batches.csv
├── calculations/
│   └── LEVEL/MOLECULE/STATE/
│       ├── molecule.xyz
│       ├── input.inp
│       └── metadata.json
└── jobs/
    ├── batch_0001/
    │   ├── calculations.csv
    │   ├── queue.csv
    │   ├── dispatcher.py
    │   └── run.sh
    └── ...
```

`prepare` does **not** submit anything, which makes it possible to inspect all generated ORCA
inputs and Slurm scripts before spending cluster resources.

If an existing prepared project already contains ORCA outputs and you only want to regenerate
the scheduler files (for example after upgrading from 1.0.x to 1.1), use:

```bash
pka-calculator rebuild-jobs run_001
```

This replaces only `jobs/` and `batches.csv`; `calculations/` and existing outputs are preserved.
Do not run it while old Slurm jobs from that project are still active.

## 5. Submit grouped jobs

```bash
pka-calculator submit run_001
```

A grouped job behaves conceptually as:

```text
one Slurm allocation (one node)
├── worker 0 -> 4 physical cores -> calc -> calc -> calc -> ...
├── worker 1 -> 4 physical cores -> calc -> calc -> calc -> ...
├── worker 2 -> 4 physical cores -> calc -> calc -> calc -> ...
└── shared pending queue
```

When one calculation finishes, that worker immediately takes the next pending calculation;
it does not wait for the other workers. Before the queue starts, the dispatcher performs a
short OpenMPI preflight and verifies that each worker's MPI ranks remain inside that worker's
physical-core set.

Each calculation writes its own:

```text
output.out
output.err
exit_code.txt
wall_seconds.txt
node.txt
job_id.txt
worker.txt
```

Each batch additionally writes `status.csv`, `events.csv`, `summary.txt`, preflight logs, and
`cpu_topology.json`. Slurm sends `TERM` shortly before walltime; the batch shell forwards it
to the dispatcher, which terminates active ORCA process groups cleanly and leaves unfinished
work resumable.

Calling `submit` again is restart-safe: already submitted batches are skipped unless
`--resubmit` is explicitly supplied. With `--resubmit`, already successful calculations are
skipped by the dispatcher and only incomplete/failed calculations are run again.

## 6. Monitor and collect

```bash
pka-calculator status run_001
pka-calculator collect run_001
```

`collect` creates `run_001/results.csv` with, where available:

- ORCA version
- normal termination
- SCF/optimization status
- final single-point energy
- Gibbs free energy
- enthalpy
- zero-point energy
- thermal Gibbs correction
- temperature
- imaginary-frequency count / minimum frequency
- ORCA and shell wall time
- exit code and compute node
- `valid` flag and validation errors

By default a calculation requesting `OPT + Freq` is not considered valid when:

- ORCA did not terminate normally;
- an SCF failure is detected;
- optimization convergence is not confirmed;
- Gibbs free energy is absent;
- a frequency is below `thermodynamics.imaginary_frequency_tolerance_cm1`.

## 7. Thermodynamic routes

At temperature `T`, values are calculated using `RT ln(10)`.

### Acid dissociation

```text
HA -> A- + H+
```

```text
pKa = [G(A-) + G(H+) - G(HA)] / [RT ln(10)]
```

Route name: `acid`.

### pKa of a conjugate acid

```text
BH+ -> B + H+
```

```text
pKa(BH+) = [G(B) + G(H+) - G(BH+)] / [RT ln(10)]
```

Route name: `conjugate_acid`.

For several protonated states, the lowest-free-energy `BH+` state is selected. This
corresponds to the **largest** site-specific conjugate-acid pKa, not the smallest pKa.

### Direct pKb

```text
B + H2O -> BH+ + OH-
```

The package stores an effective reference

```text
Gref = G(OH-) - G(H2O)
```

and uses

```text
pKb = [G(BH+) + Gref - G(B)] / [RT ln(10)]
```

Route name: `pkb`.

Alternatively:

```text
pKb = pKw - pKa(BH+)
```

Route name: `pkb_from_pka`. Configure `pkw` for the temperature/model you intend to use;
`14.0` is only the default starter value.

### Route-oriented energy descriptors

For empirical linear calibration without an explicit reference species, use the public
route-aware descriptor API instead of a generic `ion - neutral` difference:

```python
from pka_calculator import reaction_energy_table

microstates = reaction_energy_table(results, route="conjugate_acid")
# microstates["reaction_energy_kj_mol"] = G(B) - G(BH+)
# microstates["selected"] marks the maximum micro-pKaH branch.
```

The reference-free descriptor omits `G(H+)` for pKa/pKaH and
`G(OH-) - G(H2O)` for direct pKb. Those terms are constant within a calculation
level and are absorbed by the calibration intercept. Do not fit pKaH against
`G(BH+) - G(B)`: that has the sign of direct pKb and produces an artificial
negative pKaH coefficient.

## 8. Fit the reference on a calibration set

Experimental file may use modern columns:

```csv
molecule_id,pka_exp
1,4.76
2,3.21
```

The old `Molecule; pKa (exp)` naming is also recognized.

For acid pKa:

```bash
pka-calculator calibrate run_001 calibration.csv \
  --route acid \
  -o proton_reference.csv
```

For conjugate-acid pKa:

```bash
pka-calculator calibrate run_001 calibration.csv \
  --route conjugate_acid \
  -o proton_reference.csv
```

For direct pKb:

```bash
pka-calculator calibrate run_001 calibration_pkb.csv \
  --route pkb \
  -o hydroxide_reference.csv
```

The command writes both the fitted reference and `*_training_details.csv`, so the exact
molecules contributing to the fit remain auditable.

**Do not evaluate final MAE/RMSE on the same molecules used to fit the reference or a linear
post-calibration if you want an unbiased test error.**

## 9. Predict

```bash
pka-calculator predict run_001 proton_reference.csv \
  --route acid \
  -o predictions.csv
```

`predictions.csv` contains every site plus a `selected` flag for the thermodynamically
selected state.

Conjugate-acid pKa:

```bash
pka-calculator predict run_001 proton_reference.csv \
  --route conjugate_acid \
  -o pka_bh.csv
```

pKb from pKa:

```bash
pka-calculator predict run_001 proton_reference.csv \
  --route pkb_from_pka --pkw 14.0 \
  -o pkb.csv
```

## 10. Independent metrics and plots

```bash
pka-calculator metrics predictions.csv test_set.csv -o metrics.csv
pka-calculator plot predictions.csv test_set.csv plots/
```

Optional explicit linear calibration fit:

```bash
pka-calculator metrics calibration_predictions.csv calibration.csv \
  --fit-calibration -o calibration_metrics.csv
```

Fit the linear correction on a calibration/training set, then apply it separately in your
analysis code to an independent test set.

Interactive HTML:

```bash
pka-calculator interactive predictions.csv test_set.csv -o report.html
```

## 11. Extract optimized geometries

After collection:

```bash
pka-calculator extract-geometries run_001 optimized_xyz/ \
  --level PBE0_6-31+Gstar
```

The last frame of `input_trj.xyz` is used when present, otherwise the starting XYZ is copied.
Invalid calculations are excluded unless `--include-invalid` is requested.

## Library API

```python
import pandas as pd
from pka_calculator.thermodynamics import calibrate_reference, predict_acid_base

results = pd.read_csv("run_001/results.csv")
calibration = pd.DataFrame({
    "molecule_id": ["1", "2"],
    "experimental": [4.76, 3.21],
})

references, training_details = calibrate_reference(
    results,
    calibration,
    route="acid",
    temperature_k=298.15,
)

predictions = predict_acid_base(
    results,
    references,
    route="acid",
    temperature_k=298.15,
)
```

## Tests

```bash
pytest
```

The initial test suite covers:

- grouped-core packing without oversubscription;
- ORCA input generation;
- ORCA runtime/energy/frequency parsing and validation;
- recovery of a known proton reference from calibration data;
- route-specific selection of acid, conjugate-acid, and pKb states;
- migration of old neutral/protonated/deprotonated XYZ naming.

## Design notes

See [ARCHITECTURE.md](ARCHITECTURE.md) and [MIGRATION.md](MIGRATION.md).

## v1.2: explicit acid/base semantics and train-only linear calibration

Thermodynamic routes are chemically explicit:

- `acid`: `HA -> A- + H+`
- `conjugate_acid`: `BH+ -> B + H+`
- `pkb`: `B + H2O -> BH+ + OH-`
- `pkb_from_pka`: `pKb(B) = pKw - pKa(BH+)`

For a neutral carboxylic acid used as the base `B = HA`, the pKb calculation
therefore requires the **protonated** state `H2A+`.  The deprotonated carboxylate
`A-` belongs to the acid-pKa route and is not silently reused as the base state.
When charge metadata is available, route validation rejects inconsistent state
labels (for example a `neutral` row with charge `-1`).

RDKit state generation now records formal charge from the generated molecular
graph instead of hard-coding state charges.

Linear correction is fitted in the directly applicable direction

```text
experimental = slope * calculated + intercept
```

using an explicit training table only.  Python API:

```python
from pka_calculator.analysis import (
    fit_linear_calibration,
    apply_linear_calibration,
)

fit = fit_linear_calibration(train_predictions, train_experimental)
all_corrected = apply_linear_calibration(all_predictions, fit)
```

CLI:

```bash
pka-calculator linear-calibrate predictions.csv train.csv \
    --output-calibration linear_calibration.csv \
    --output-predictions predictions_linear_calibrated.csv
```

The calibration coefficients must be frozen before evaluating the independent
test subset.
