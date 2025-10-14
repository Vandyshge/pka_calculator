import os
import re
import csv
from pathlib import Path
import pprint
import json

def parse_output_file(output_file):
    gibbs_energy = None
    run_time_min = None
    
    with open(output_file, 'r') as f:
        content = f.read()
        
        gibbs_match = re.search(r'Final Gibbs free energy\s+\.\.\.\s+([-\d\.]+)\s+Eh', content)
        if gibbs_match:
            gibbs_energy = float(gibbs_match.group(1))
        
        time_match = re.search(r'TOTAL RUN TIME:\s+(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes', content)
        if time_match:
            days = int(time_match.group(1))
            hours = int(time_match.group(2))
            minutes = int(time_match.group(3))
            run_time_min = days*24*60 + hours*60 + minutes
    
    return gibbs_energy, run_time_min

def collect_results(calc_dir):
    results = {}
    
    for root, dirs, files in os.walk(calc_dir):
        if "output.out" in files:
            output_file = os.path.join(root, "output.out")
            path_parts = Path(root).parts
            
            # calc_dir/basis/molecule/form/method
            if len(path_parts) >= 5:
                basis = path_parts[-4]
                molecule = path_parts[-3]
                form = path_parts[-2]
                method = path_parts[-1]
                
                gibbs, time = parse_output_file(output_file)
                results[(basis, molecule, method, form)] = (gibbs, time)
    
    return results

def generate_results_table(results, output_dir, name_file):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    molecules = sorted({key[1] for key in results.keys()})
    methods = sorted({key[2] for key in results.keys()})
    basis_sets = sorted({key[0] for key in results.keys()})
    
    base_molecules = {}
    for (basis, molecule, method, form), (gibbs, time) in results.items():
        if '_deprotonated' in molecule:
            base_name = molecule.split('_')[0]
        elif '_protonated' in molecule:
            base_name = molecule.split('_')[0]
        else:
            base_name = molecule
        
        if base_name not in base_molecules:
            base_molecules[base_name] = {}
        
        if method not in base_molecules[base_name]:
            base_molecules[base_name][method] = {}
        
        if basis not in base_molecules[base_name][method]:
            base_molecules[base_name][method][basis] = {'neutral': None, 'deprotonated': {}, 'protonated': {}}
        
        if form == "neutral":
            base_molecules[base_name][method][basis]['neutral'] = (gibbs, time)
        elif "deprotonated" in form:
            base_molecules[base_name][method][basis]['deprotonated'][form] = (gibbs, time)
        elif "protonated" in form:
            base_molecules[base_name][method][basis]['protonated'][form] = (gibbs, time)
    
    headers = ["Molecule", "Method", "Basis", "Calculation_Form"]
    energy_headers = []
    time_headers = []
    for form_type in ["N", "D", "P"]:
        energy_headers.extend([f"G_{form_type}"])
        time_headers.extend([f"t_{form_type}"])
    
    headers.extend(energy_headers)
    headers.extend(time_headers)
    
    rows = []
    for base_name in sorted(base_molecules.keys()):
        for method in sorted(base_molecules[base_name].keys()):
            for basis in sorted(base_molecules[base_name][method].keys()):
                method_data = base_molecules[base_name][method][basis]
                
                all_forms = set()
                all_forms.update(method_data['deprotonated'].keys())
                all_forms.update(method_data['protonated'].keys())
                
                if not all_forms and method_data['neutral']:
                    gibbs_n, time_n = method_data['neutral']
                    row = [
                        base_name,  # Molecule
                        method,     # Method
                        basis,      # Basis
                        "neutral",  # Calculation_Form
                        f"{gibbs_n:.6f}" if gibbs_n is not None else '',  # G_N
                        '',  # G_D
                        '',  # G_P
                        str(time_n) if time_n is not None else '',  # t_N
                        '',  # t_D
                        ''   # t_P
                    ]
                    rows.append(row)
                
                for form in sorted(all_forms):
                    form_type = "deprotonated" if "deprotonated" in form else "protonated"
                    form_suffix = form.replace('_deprotonated', '').replace('_protonated', '')
                    full_molecule_name = f"{base_name}_{form_suffix}"
                    
                    calculation_form = "deprotonated" if "deprotonated" in form else "protonated"
                    
                    gibbs_n, time_n = method_data['neutral'] if method_data['neutral'] else (None, None)
                    
                    if form_type == "deprotonated" and form in method_data['deprotonated']:
                        gibbs_d, time_d = method_data['deprotonated'][form]
                    else:
                        gibbs_d, time_d = (None, None)
                    
                    if form_type == "protonated" and form in method_data['protonated']:
                        gibbs_p, time_p = method_data['protonated'][form]
                    else:
                        gibbs_p, time_p = (None, None)

                    row = [
                        full_molecule_name,  # Molecule
                        method,              # Method
                        basis,               # Basis
                        calculation_form,    # Calculation_Form
                        f"{gibbs_n:.6f}" if gibbs_n is not None else '',  # G_N
                        f"{gibbs_d:.6f}" if gibbs_d is not None else '',  # G_D
                        f"{gibbs_p:.6f}" if gibbs_p is not None else '',  # G_P
                        str(time_n) if time_n is not None else '',  # t_N
                        str(time_d) if time_d is not None else '',  # t_D
                        str(time_p) if time_p is not None else ''   # t_P
                    ]
                    rows.append(row)
    
    csv_file = output_dir / f"results_{name_file}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        writer.writerows(rows)
    
    return csv_file

def process_results(calc_dir, output_dir, name_file):
    """Process calculation results and generate output files"""
    print(f"Processing results from {calc_dir}")
    
    results = collect_results(calc_dir)
    csv_file = generate_results_table(results, output_dir, name_file)
    
    print(f"Results processed successfully! Output saved to {csv_file}")