import pandas as pd
import numpy as np
from pathlib import Path

# Constants
R = 8.314462618  # J/(mol·K)
T = 298.15  # K
RT_in10 = R * T * np.log(10) / 1000  # kJ/mol
HARTREE_TO_KJ = 2625.5

def calculate_gh_for_molecule_deprotonated(molecule_data):
    """Calculate G(H+) for a molecule using only the form with minimal g_a - g_ha"""
    molecule_data['delta_g'] = molecule_data['G_D'] - molecule_data['G_N']
    min_idx = molecule_data['delta_g'].idxmin()
    row = molecule_data.loc[min_idx]
    g_ha = row['G_N'] * HARTREE_TO_KJ
    g_a = row['G_D'] * HARTREE_TO_KJ
    pka = row['pKa (exp)']
    g_h = pka * RT_in10 + g_ha - g_a
    return g_h, min_idx, row['Molecule']

def calculate_gh_for_molecule_protonated(molecule_data):
    """Calculate G(H+) for a molecule using protonated form with minimal g_ha - g_h2a"""
    molecule_data['delta_g'] = molecule_data['G_N'] - molecule_data['G_P']
    min_idx = molecule_data['delta_g'].idxmin()
    row = molecule_data.loc[min_idx]
    g_ha = row['G_N'] * HARTREE_TO_KJ
    g_h2a = row['G_P'] * HARTREE_TO_KJ
    pka = row['pKa (exp)']
    g_h = pka * RT_in10 + g_h2a - g_ha
    return g_h, min_idx, row['Molecule']

def calculate_pka_deprotonated(g_ha, g_a, g_h):
    delta_g = g_a + g_h - g_ha
    return delta_g / RT_in10

def calculate_pka_protonated(g_ha, g_h2a, g_h):
    delta_g = g_ha + g_h - g_h2a
    return delta_g / RT_in10

def analyze_results(results_dir, experimental_file, output_dir, name_file):
    print(f"Analyzing results from {results_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results_df = pd.read_csv(Path(results_dir) / f"results_{name_file}.csv", sep=';')
    exp_df = pd.read_csv(experimental_file, sep=';')[['Molecule', 'pKa (exp)']].dropna()

    results_df['Molecule'] = results_df['Molecule'].astype(str)
    exp_df['Molecule'] = exp_df['Molecule'].astype(str)

    results_df['Base_Molecule'] = results_df['Molecule'].str.split('_').str[0]
    exp_df['Base_Molecule'] = exp_df['Molecule'].str.split('_').str[0]

    merged_df = pd.merge(results_df, exp_df[['Base_Molecule', 'pKa (exp)']],
                         on='Base_Molecule', how='left')

    gh_entries = []

    for (method, basis), group in merged_df.groupby(["Method", "Basis"]):
        # deprotonated
        deprot = group.dropna(subset=['G_N', 'G_D', 'pKa (exp)']).copy()
        for base_mol, mol_group in deprot.groupby("Base_Molecule"):
            try:
                gh, idx, sel_mol = calculate_gh_for_molecule_deprotonated(mol_group)
                gh_entries.append({
                    "Method": method, "Basis": basis, "Calculation_Form": "deprotonated",
                    "Base_Molecule": base_mol, "Selected_Molecule": sel_mol,
                    "G(H+) (kJ/mol)": gh
                })
            except: 
                pass

        # protonated
        prot = group.dropna(subset=['G_N', 'G_P', 'pKa (exp)']).copy()
        for base_mol, mol_group in prot.groupby("Base_Molecule"):
            try:
                gh, idx, sel_mol = calculate_gh_for_molecule_protonated(mol_group)
                gh_entries.append({
                    "Method": method, "Basis": basis, "Calculation_Form": "protonated",
                    "Base_Molecule": base_mol, "Selected_Molecule": sel_mol,
                    "G(H+) (kJ/mol)": gh
                })
            except:
                pass

    gh_df = pd.DataFrame(gh_entries)
    gh_df.to_csv(output_dir / f"gh_values_{name_file}.csv", sep=";", index=False)

    gh_stats = (
        gh_df.groupby(["Method", "Basis", "Calculation_Form"])
             .agg(
                 Mean=("G(H+) (kJ/mol)", "mean"),
                 Std=("G(H+) (kJ/mol)", "std"),
                 Min=("G(H+) (kJ/mol)", "min"),
                 Max=("G(H+) (kJ/mol)", "max"),
                 N=("G(H+) (kJ/mol)", "count")
             )
             .reset_index()
    )
    gh_stats.to_csv(output_dir / f"gh_stats_{name_file}.csv", sep=";", index=False)

    merged_df["pKa_calc"] = np.nan
    
    for idx, row in merged_df.iterrows():
        gh_candidates = gh_df[
            (gh_df["Method"] == row.Method) &
            (gh_df["Basis"] == row.Basis) &
            (gh_df["Calculation_Form"] == row.Calculation_Form)
        ]
        if gh_candidates.empty:
            continue
    
        gh_mean = gh_candidates["G(H+) (kJ/mol)"].mean()
    
        if row.Calculation_Form == "deprotonated" and pd.notna(row.G_N) and pd.notna(row.G_D):
            g_ha = row.G_N * HARTREE_TO_KJ
            g_a  = row.G_D * HARTREE_TO_KJ
            merged_df.at[idx, "pKa_calc"] = calculate_pka_deprotonated(g_ha, g_a, gh_mean)
    
        elif row.Calculation_Form == "protonated" and pd.notna(row.G_N) and pd.notna(row.G_P):
            g_ha  = row.G_N * HARTREE_TO_KJ
            g_h2a = row.G_P * HARTREE_TO_KJ
            merged_df.at[idx, "pKa_calc"] = calculate_pka_protonated(g_ha, g_h2a, gh_mean)
            
    merged_df.to_csv(output_dir / f"pka_{name_file}.csv", sep=";", index=False)
    print(f"Saved pKa results to {output_dir}/pka_{name_file}.csv")