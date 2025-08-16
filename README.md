# $pK_a$ Calculator

Python library for calculating pKa values using quantum chemical methods

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

- Python 3.8+
- ORCA (must be installed and in PATH)
- Slurm (for cluster execution)
- pandas
- numpy
- matplotlib
- scipy

## Usage example

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

or 6. Run the results through the processing pipeline:
```bash
pka-calculator pipeline mycalculations/ -e experimental_pka.csv -o data
```

7. Calculate pKa values for neutral and deprotonated species (stepwise approach):
```bash
# Step 1: Calculate neutral molecules
pka-calculator calculate molecules_neutral/ -b "6-31+G*" -m PBE -o mycalculations -f neutral

# Step 2: Generate deprotonated structures
pka-calculator deprotonate mycalculations/ -o molecules_deprotonated/

# Step 3: Calculate deprotonated species
pka-calculator calculate molecules_deprotonated/ -b "6-31+G*" -m PBE -o mycalculations -f deprotonated
```

## File Structure
```txt
pka_calculator/
├── pka_calculator/                   # Core code
│   ├── cli.py                        # Command Line Interface
│   ├── calculator.py                 # Run calculations
│   ├── monitor.py                    # Job monitoring
│   ├── deprotonator.py               # Deprotonated molecules
│   ├── processor.py                  # Process results
│   ├── analyzer.py                   # Data analysis  
│   ├── visualizer.py                 # Visualization
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
