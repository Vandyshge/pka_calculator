# pKa Calculator

Python library for calculating pKa values using quantum chemical methods

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/pka_calculator.git
cd pka_calculator
```

3. Install in development mode:
```bash
pip install -e .
```

5. Checking the installation
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

## Basic Usage

1. Run calculations:
```bash
pka-calculator calculate molecules/ -b "6-31+G*" -m PBE -o calculations
```

3. Monitor jobs:
```bash
pka-calculator monitor calculations_summary.txt -u yourusername
```

5. Process results:
```bash
pka-calculator process calculations/ -o results
```

7. Analyze with experimental data:
```bash
pka-calculator analyze results/ -e experimental_pka.csv -o analysis
```

9. Visualize results:
```bash
pka-calculator visualize analysis/ -o plots
```

## File Structure
```txt
pka_calculator/
├── pka_calculator/       # Core code
│   ├── calculator.py     # Run calculations
│   ├── processor.py      # Process results
│   ├── analyzer.py       # Data analysis  
│   ├── visualizer.py     # Visualization
│   ├── monitor.py        # Job monitoring
│   └── __init__.py
├── pyproject.toml        # Setup file
└── README.md             # Documentation
```
