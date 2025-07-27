pKa Calculator
==============

Python library for calculating pKa values using quantum chemical methods

Installation
------------
1. Clone repository:
git clone https://github.com/yourusername/pka_calculator.git
cd pka_calculator

2. Install in development mode:
pip install -e .

3. Checking the installation
pka-calculator --help

Dependencies
------------
- Python 3.8+
- ORCA (must be installed and in PATH)
- Slurm (for cluster execution)
- pandas
- numpy
- matplotlib
- scipy

Basic Usage
-----------
1. Run calculations:
pka-calculator calculate molecules/ -b "6-31+G*" -m PBE -o calculations

2. Monitor jobs:
pka-calculator monitor calculations_summary.txt -u yourusername

3. Process results:
pka-calculator process calculations/ -o results

4. Analyze with experimental data:
pka-calculator analyze results/ -e experimental_pka.csv -o analysis

5. Visualize results:
pka-calculator visualize analysis/ -o plots

File Structure
--------------
pka_calculator/
├── pka_calculator/       # Core code
│   ├── calculator.py     # Run calculations
│   ├── processor.py      # Process results
│   ├── analyzer.py       # Data analysis  
│   ├── visualizer.py     # Visualization
│   ├── monitor.py        # Job monitoring
│   └── __init__.py
├── pyproject             # Setup file
└── README.md             # Documentation
