# $pK_a$ Calculator

Python library for calculating pKa values using quantum chemical methods

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/pka_calculator.git
cd pka_calculator
```

2. Install in development mode:
```bash
pip install -e .
```

3. Checking the installation
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

0. Go to folder
```bash
cd example
```

2. Run calculations:
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

## File Structure
```txt
pka_calculator/
├── pka_calculator/          # Core code
│   ├── calculator.py        # Run calculations
│   ├── processor.py         # Process results
│   ├── analyzer.py          # Data analysis  
│   ├── visualizer.py        # Visualization
│   ├── monitor.py           # Job monitoring
│   └── __init__.py      
├── example/                 # Example
│   ├── molecules/           # .xyz files
|   ├── solutions/           # Solutions
│   └── experimental_pka.csv # Experimental data
├── pyproject.toml           # Setup file
└── README.md                # Documentation
```
