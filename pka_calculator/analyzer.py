import pandas as pd
import numpy as np
import csv
from pathlib import Path

# Constants
R = 8.314462618  # J/(mol·K)
T = 298.15  # K
RT_in10 = R * T * np.log(10) / 1000  # kJ/mol

def calculate_gh(method_results):
    """Calculate G(H+) for a method"""
    g_ha = method_results['G_N'] * 2625.5  # Eh to kJ/mol
    g_a = method_results['G_D'] * 2625.5
    pka = method_results['pKa (exp)']
    
    g_h = pka * RT_in10 + g_ha - g_a
    return g_h

def calculate_pka(g_ha, g_a, g_h):
    """Calculate pKa from Gibbs energies"""
    delta_g = g_a + g_h - g_ha
    return delta_g / RT_in10

def analyze_results(results_dir, experimental_file, output_dir, name_file):
    """Analyze results and calculate pKa values"""
    print(f"Analyzing results from {results_dir}")
    print(f"Using experimental data from {experimental_file}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    results_df = pd.read_csv(Path(results_dir) / f"results_{name_file}.csv", sep=';')
    pka_df = pd.read_csv(experimental_file, sep=';')
    pka_df = pka_df[['Acids', 'pKa (exp)']].dropna()
    
    # Merge data
    merged_df = pd.merge(results_df, pka_df, left_on='Molecule', right_on='Acids')
    
    # Calculate G(H+) for each method
    methods = [col.split('_')[0] for col in results_df.columns if '_G_N' in col]
    gh_results = {}
    
    for method in methods:
        method_df = merged_df[[
            'Molecule', 'pKa (exp)',
            f'{method}_G_N', f'{method}_G_D'
        ]].copy()
        method_df.columns = ['Molecule', 'pKa (exp)', 'G_N', 'G_D']
        
        g_h = calculate_gh(method_df)
        gh_mean = np.mean(g_h)
        gh_results[method] = {
            'mean': gh_mean,
            'std': np.std(g_h, ddof=1),
            'max': np.max(g_h),
            'min': np.min(g_h)
        }
        
        # Calculate pKa values
        g_ha = method_df['G_N'] * 2625.5
        g_a = method_df['G_D'] * 2625.5
        merged_df[f'pKa_{method}'] = calculate_pka(g_ha, g_a, gh_mean)
    
    # Save G(H+) results
    gh_file = output_dir / f"gh_{name_file}.csv"
    with open(gh_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Method', 'Mean G(H+) (kJ/mol)', 'Std G(H+) (kJ/mol)',
                        'Max G(H+) (kJ/mol)', 'Min G(H+) (kJ/mol)'])
        for method, values in gh_results.items():
            writer.writerow([
                method, values['mean'], values['std'],
                values['max'], values['min']
            ])
    
    # Save pKa results
    pka_file = output_dir / f"pka_{name_file}.csv"
    merged_df.to_csv(pka_file, index=False, sep=';')
    
    print(f"Analysis complete! Results saved to {output_dir}")
