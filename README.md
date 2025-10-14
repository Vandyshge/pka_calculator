# $pK_a$ Calculator

Python library for calculating and analyzing pKa values using quantum chemical methods and automated pipelines.

## Installation

1. Clone repository:

```bash
git clone https://github.com/Vandyshge/pka_calculator.git
cd pka_calculator
```

2. Install in development mode:

```bash
pip install -e .
```

3. Checking the installation:

```bash
pka-calculator --help
```

## Dependencies

* Python 3.8+
* ORCA (must be installed and in PATH)
* Slurm (for cluster execution)
* pandas
* numpy
* matplotlib
* scipy
* plotly

---

## Usage Example

### Basic Workflow

0. Go to folder:

```bash
cd example
```

1. Run calculations:

```bash
pka-calculator calculate molecules/ -b "6-31+G*" -m PBE -o mycalculations
```

2. Monitor jobs:

```bash
pka-calculator monitor mycalculations -u yourusername
```

3. Process results:

```bash
pka-calculator process mycalculations/ -o results
```

4. Analyze with experimental data:

```bash
pka-calculator analyze results/ -e experimental_pka.csv -o analysis
```

5. Visualize results:

```bash
pka-calculator visualize analysis/ -o plots
```

6. (Optional) Run full processing pipeline:

```bash
pka-calculator pipeline mycalculations/ -e experimental_pka.csv -o data -n basis
```

---

### Stepwise pKa Calculation

```bash
# Step 1: Calculate neutral molecules
pka-calculator calculate molecules_neutral/ -b "6-31+G*" -m PBE -o mycalculations -f neutral

# Step 2: Generate deprotonated structures
pka-calculator deprotonate mycalculations/ -o molecules_deprotonated/

# Step 3: Calculate deprotonated species
pka-calculator calculate molecules_deprotonated/ -b "6-31+G*" -m PBE -o mycalculations -f deprotonated
```

---

## Additional CLI Commands

### Deprotonation

Create deprotonated molecules interactively from optimized structures:

```bash
pka-calculator deprotonate mycalculations/ -o molecules_deprotonated/
```

### Equilibration

Extract equilibrated (last-frame) structures from trajectories:

```bash
pka-calculator equilibrate mycalculations/ -o equilibrated/
```

### Interactive HTML Visualization

Generate interactive plots of experimental vs calculated pKa and molecule forms:

```bash
pka-calculator interactive analysis/ -n basis -o html/
```

### Extract Minimum pKa

Generate CSV file containing minimal pKa for each molecule:

```bash
pka-calculator minpka analysis/ -o results/ -n min
```

---

## Full Pipeline

Run complete data processing, analysis, visualization, and HTML report generation:

```bash
pka-calculator pipeline mycalculations/ \
  -e experimental_pka.csv \
  -o analysis \
  -n basis
```

This command will:

1. Process calculation results (`process`)
2. Analyze data and compute pKa values (`analyze`)
3. Extract minimal pKa per molecule (`minpka`)
4. Generate plots (`visualize`)
5. Build an interactive HTML report (`interactive`)

---

## Output Files

| File                         | Description                               |
| ---------------------------- | ----------------------------------------- |
| `results_*.csv`              | Parsed energies from ORCA outputs         |
| `gh_values_*.csv`            | G(H⁺) estimates for each method/basis     |
| `gh_stats_*.csv`             | Summary statistics of G(H⁺)               |
| `pka_*.csv`                  | Calculated pKa values                     |
| `pka_min_*.csv`              | Minimal pKa per molecule                  |
| `pka_min_*.png`              | Comparison plots (Exp vs Calc)            |
| `pka_*_interactive.html`     | Interactive visualization                 |
| `removed_hydrogens.csv`      | Info on removed hydrogens (deprotonation) |
| `equilibrated_molecules.csv` | Info on equilibrated structures           |

---

## Running on a Cluster

The tool uses **SLURM** for parallel ORCA job submission.
Ensure `orca` and `sbatch` are available in your PATH.

Each calculation group script (`orca_group_*.sh`) is generated and submitted automatically.
Adjust `--tasks-per-node` to control load per node.

---

## File Structure

```txt
pka_calculator/
├── pka_calculator/                   # Core code
│   ├── cli.py                        # Command Line Interface
│   ├── calculator.py                 # Run calculations
│   ├── monitor.py                    # Job monitoring
│   ├── deprotonator.py               # Deprotonated molecules
│   ├── equilibrator.py               # Equilibrated XYZ files
│   ├── processor.py                  # Process results
│   ├── analyzer.py                   # Data analysis  
│   ├── visualizer.py                 # Visualization
│   ├── interactive.py                # Interactive HTML generation
│   ├── min_pka.py                    # Extract minimal pKa
│   └── __init__.py      
├── example/                          # Example
│   ├── molecules/                    # .xyz files
│   │   ├── molecule.xyz              # Molecule
|   |   └── molecule_deprotonated.xyz # Deprotonated molecule
|   ├── solutions/                    # Solutions
│   └── experimental_pka.csv          # Experimental data
├── pyproject.toml                    # Setup file
├── .gitignore                        # Gitignore
└── README.md                         # Documentation
```
