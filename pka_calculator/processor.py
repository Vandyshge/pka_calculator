import os
import re
import csv
from pathlib import Path

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
            molecule = path_parts[-3]  # calc_dir/basis/molecule/form/method
            form = path_parts[-2]
            method = path_parts[-1]
            
            gibbs, time = parse_output_file(output_file)
            results[(molecule, method, form)] = (gibbs, time)
    
    return results

def generate_latex_table(results, name_file):
    latex = f"""\\begin{{tabular}}{{lcccccc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{HF}} & \\multicolumn{{2}}{{c}}{{B3LYP}} & \\multicolumn{{2}}{{c}}{{PBE0}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
\\multirow{{2}}{{*}}{{Molecule}} & G$_N$ & G$_D$ & G$_N$ & G$_D$ & G$_N$ & G$_D$ \\\\
 & t$_N$ & t$_D$ & t$_N$ & t$_D$ & t$_N$ & t$_D$ \\\\
\\midrule
"""

    molecules = sorted({key[0] for key in results.keys()})
    methods = sorted({key[1] for key in results.keys()})
    
    for mol in molecules:
        energy_row = f"\\multirow{{2}}{{*}}{{{mol}}}"
        time_row = " "
        
        for method in methods:
            for form in ["neutral", "deprotonated"]:
                key = (mol, method, form)
                if key in results:
                    gibbs, time = results[key]
                    energy_row += f" & {gibbs:.6f}" if gibbs is not None else " & --"
                    time_row += f" & {time}" if time is not None else " & --"
                else:
                    energy_row += " & --"
                    time_row += " & --"
        
        latex += energy_row + " \\\\\n" + time_row + " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}"""
    
    return latex

def generate_results_table(results, output_dir, name_file):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    molecules = sorted({key[0] for key in results.keys()})
    methods = sorted({key[1] for key in results.keys()})
    
    headers = ["Molecule"]
    for method in methods:
        headers.extend([
            f"{method}_G_N",
            f"{method}_G_D",
            f"{method}_t_N",
            f"{method}_t_D"
        ])
    
    rows = []
    for mol in molecules:
        row = [mol]
        
        for method in methods:
            key_n = (mol, method, "neutral")
            gibbs_n, time_n = results.get(key_n, (None, None))
            
            key_d = (mol, method, "deprotonated")
            gibbs_d, time_d = results.get(key_d, (None, None))
            
            row.extend([
                f"{gibbs_n:.6f}" if gibbs_n is not None else '',
                f"{gibbs_d:.6f}" if gibbs_d is not None else '',
                str(time_n) if time_n is not None else '',
                str(time_d) if time_d is not None else ''
            ])
        
        rows.append(row)
    
    csv_file = output_dir / f"results_{name_file}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        writer.writerows(rows)
    
    latex_table = generate_latex_table(results, name_file)
    tex_file = output_dir / f"results_{name_file}.tex"
    with open(tex_file, 'w') as f:
        f.write(latex_table)
    
    return csv_file, tex_file

def process_results(calc_dir, output_dir, name_file):
    """Process calculation results and generate output files"""
    print(f"Processing results from {calc_dir}")
    
    results = collect_results(calc_dir)
    csv_file, tex_file = generate_results_table(results, output_dir, name_file)
    
    print(f"Results processed successfully! Output saved to {csv_file} and {tex_file}")