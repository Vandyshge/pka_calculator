import os
import subprocess
from pathlib import Path
import re

METHOD_TEMPLATES = {
    'HF': "! HF {basis} TightSCF CPCM(water) OPT Freq",
    'B3LYP': "! B3LYP {basis} TightSCF CPCM(water) OPT Freq",
    'B3LYP-D3-Gdp': "! B3LYP D3 6-311G(d,p) TightSCF CPCM(water) OPT Freq",
    'B3LYP-D3-Gdfp': "! B3LYP D3 6-311+G(2df,2p) TightSCF CPCM(water) OPT Freq",
    'B3LYP-D3': "! B3LYP D3 {basis} TightSCF CPCM(water) OPT Freq",
    'PBE0': "! PBE0 {basis} TightSCF CPCM(water) OPT Freq",
    'PBE': "! PBE {basis} TightSCF CPCM(water) OPT Freq",
    'M062X': "! M062X {basis} TightSCF CPCM(water) OPT Freq",
    # 'DLPNO-CCSDT': "! DLPNO-CCSD(T) {basis} TightSCF CPCM(water) Freq", # no optimizing 
    'DLPNO-CCSDT': "! DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C TIGHTSCF CPCM(water) NumFreq", # no optimizing 
    'MP2': "! MP2 {basis} TightSCF CPCM(water) OPT Freq",
    'LSDA': "! LDA {basis} TightSCF CPCM(water) OPT Freq",
    'CCSDT': "! CCSD(T) {basis} TightSCF CPCM(water) Freq",
    'AM1': "! AM1 {basis} TightSCF CPCM(water) OPT Freq",
    'WB97X-D3': "! WB97X-D3 {basis} TightSCF CPCM(water) OPT Freq",
    'WB97X-D3-Gdp': "! WB97X-D3 6-311G(d,p) TightSCF CPCM(water) OPT Freq",
    'HF-Gdp': "! HF 6-311G(d,p) TightSCF CPCM(water) OPT Freq",
    'CAM-B3LYP': "! CAM-B3LYP {basis} TightSCF CPCM(water) OPT Freq",
    'PM3': "! PM3 TightSCF CPCM(water) OPT Freq",
    'B3LYP-D3-Gdfp': "! B3LYP D3 6-311+G(2df,2p) TightSCF CPCM(water) OPT Freq",
}

# 6-311+G(2df,p) 6-311+G2dfp
# 6-311G(d,p)
# 6-31+G*

# METHOD_TEMPLATES = {
#     'HF': "! HF {basis} TightSCF OPT Freq",
#     'B3LYP': "! B3LYP {basis} TightSCF OPT Freq",
#     'B3LYP-D3-Gdp': "! B3LYP D3 6-311G(d,p) TightSCF OPT Freq",
#     'B3LYP-D3': "! B3LYP D3 {basis} TightSCF OPT Freq",
#     'PBE0': "! PBE0 {basis} TightSCF OPT Freq",
#     'PBE': "! PBE {basis} TightSCF OPT Freq",
#     'M062X': "! M062X {basis} TightSCF OPT Freq",
#     'DLPNO-CCSDT': "! DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C TIGHTSCF NumFreq", # no optimizing 
#     'MP2': "! MP2 {basis} TightSCF OPT Freq",
#     'LSDA': "! LDA {basis} TightSCF OPT Freq",
#     'CCSDT': "! CCSD(T) {basis} TightSCF Freq",
#     'AM1': "! AM1 {basis} TightSCF OPT Freq",
#     'WB97X-D3': "! WB97X-D3 {basis} TightSCF OPT Freq",
#     'WB97X-D3-Gdp': "! WB97X-D3 6-311G(d,p) TightSCF OPT Freq",
#     'HF-Gdp': "! HF 6-311G(d,p) TightSCF OPT Freq",
#     'CAM-B3LYP': "! CAM-B3LYP {basis} TightSCF OPT Freq",
#     'PM3': "! PM3 TightSCF OPT Freq"
# }

ELECTRON_COUNT = {
    'H': 1, 'C': 6, 'O': 8, 'N': 7,
    'B': 5, 'Br': 35, 'Si': 14, 'Cl': 17,
    'F': 9, 'Li': 2, 'S': 16,
}

def calculate_multiplicity(xyz_file, charge=0):
    total_electrons = 0
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        atoms = lines[2:]

    for line in atoms:
        if not line.strip(): continue
        symbol = line.split()[0]
        total_electrons += ELECTRON_COUNT.get(symbol, 0)

    total_electrons -= charge
    unpaired_electrons = total_electrons % 2
    multiplicity = 2 * unpaired_electrons*1/2 + 1
    return int(multiplicity)

def get_molecule_forms(xyz_dir):
    """
    Определяет все формы для каждой молекулы
    Возвращает словарь: {base_name: {'neutral': path, 'forms': {form_name: path}}}
    """
    all_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz")]
    molecules = {}
    
    for filename in all_files:
        if filename.endswith("_deprotonated.xyz"):
            base_match = re.match(r'(.+?)_([A-Za-z0-9]+)_deprotonated\.xyz', filename)
            if base_match:
                base_name = base_match.group(1)
                form_name = base_match.group(2) + "_deprotonated"
            else:
                base_name = filename.replace('_deprotonated.xyz', '')
                form_name = "deprotonated"
        elif filename.endswith("_protonated.xyz"):
            base_match = re.match(r'(.+?)_([A-Za-z0-9]+)_protonated\.xyz', filename)
            if base_match:
                base_name = base_match.group(1)
                form_name = base_match.group(2) + "_protonated"
            else:
                base_name = filename.replace('_protonated.xyz', '')
                form_name = "protonated"
        else:
            base_name = filename.replace('.xyz', '')
            form_name = "neutral"
        
        if base_name not in molecules:
            molecules[base_name] = {'neutral': None, 'forms': {}}
        
        file_path = os.path.join(xyz_dir, filename)
        if form_name == "neutral":
            molecules[base_name]['neutral'] = file_path
        else:
            molecules[base_name]['forms'][form_name] = file_path
    
    return molecules

def generate_calculations(xyz_dir, basis, methods, output_dir, forms=None, tasks_per_node=16):
    molecules = get_molecule_forms(xyz_dir)
    
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / f"calculations_summary_{basis}_{'_'.join(methods)}.csv"
    
    with open(summary_file, 'w', encoding='utf-8') as sf:
        sf.write("Molecule;Method;Basis;Form;Job ID;Status\n")
    
    all_calculations = []
    
    for base_name, molecule_data in molecules.items():
        if molecule_data['neutral']:
            for method in methods:
                if method not in METHOD_TEMPLATES:
                    continue
                    
                method_name = method
                method_dir = output_dir / basis / base_name / "neutral" / method_name
                method_dir.mkdir(parents=True, exist_ok=True)

                charge = 0
                multiplicity = calculate_multiplicity(molecule_data['neutral'], charge=charge)
                
                input_file = method_dir / "input.inp"
                with open(input_file, 'w', encoding='utf-8') as f:
                    f.write(METHOD_TEMPLATES[method].format(basis=basis) + "\n")
                    f.write(f'''
%geom  
   MaxIter 200
end
''')
                    f.write(f"* xyzfile {charge} {multiplicity} molecule.xyz\n")
             
                os.system(f"cp {molecule_data['neutral']} {method_dir}/molecule.xyz")
                
                all_calculations.append({
                    'base_name': base_name,
                    'method': method_name,
                    'basis': basis,
                    'form': "neutral",
                    'directory': method_dir
                })
                        
        for form_name, form_path in molecule_data['forms'].items():
            for method in methods:
                if method not in METHOD_TEMPLATES:
                    continue
                    
                method_name = method
                method_dir = output_dir / basis / base_name / form_name / method_name
                method_dir.mkdir(parents=True, exist_ok=True)

                if form_name == "deprotonated":
                    charge = -1
                elif form_name == "protonated":
                    charge = 1
                    
                multiplicity = calculate_multiplicity(form_path, charge=charge)
                
                input_file = method_dir / "input.inp"
                with open(input_file, 'w', encoding='utf-8') as f:
                    f.write(METHOD_TEMPLATES[method].format(basis=basis) + "\n")
                    f.write(f'''
# %cpcm
#   smd true
#   smdsolvent "water"
# end

%geom  
   MaxIter 200
end
''')
                    f.write(f"* xyzfile {charge} {multiplicity} molecule.xyz\n")
             
                os.system(f"cp {form_path} {method_dir}/molecule.xyz")
                
                all_calculations.append({
                    'base_name': base_name,
                    'method': method_name,
                    'basis': basis,
                    'form': form_name,
                    'directory': method_dir
                })
                    
    group_calculations(all_calculations, tasks_per_node, summary_file)

def group_calculations(calculations, tasks_per_node, summary_file):
    groups = []
    current_group = []
    
    for calc in calculations:
        current_group.append(calc)
        if len(current_group) >= tasks_per_node:
            groups.append(current_group)
            current_group = []
    
    if current_group:
        groups.append(current_group)
    
    for i, group in enumerate(groups):
        submit_group_job(group, i, summary_file)

def submit_group_job(group, group_id, summary_file):
    script_path = Path(f"orca_group_{group_id}.sh")
    script_content = f"""#!/bin/bash
#SBATCH --job-name=orca_group_{group_id}
#SBATCH --ntasks={len(group)}
#SBATCH --cpus-per-task=1
#SBATCH -N 1

start_time=$(date +%s)

calc_dirs=({" ".join([f'"{calc["directory"]}"' for calc in group])})

run_calculation() {{
    local task_id=$1
    local calc_dir="${{calc_dirs[$task_id]}}"
    echo "Starting calculation in: $calc_dir (task $task_id)"
    cd "$calc_dir"
    (orca input.inp > output.out 2>&1) &
    exit_code=$?
    echo "Calculation in $calc_dir completed with exit code $exit_code"
    return $exit_code
}}

export -f run_calculation
export calc_dirs

echo "Starting {len(group)} ORCA calculations on $(hostname)"
for ((i=0; i<{len(group)}; i++)); do
    run_calculation $i
done

wait
end_time=$(date +%s)
echo "All calculations in group {group_id} completed"
echo "Total execution time: $((end_time - start_time)) seconds"
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    try:
        result = subprocess.run(
            ['sbatch', str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        job_id = result.stdout.strip().split()[-1]
        status = "Submitted"
        
        with open(summary_file, 'a', encoding='utf-8') as sf:
            for calc in group:
                sf.write(f"{calc['base_name']};{calc['method']};{calc['basis']};{calc['form']};{job_id};{status}\n")

    
    except subprocess.CalledProcessError as e:
        job_id = "Failed"
        status = f"Slurm Error: {e.stderr.strip()}"
        with open(summary_file, 'a', encoding='utf-8') as sf:
            for calc in group:
                sf.write(f"{calc['base_name']};{calc['method']};{calc['basis']};{calc['form']};{job_id};{status}\n")
    except Exception as e:
        job_id = "Failed"
        status = f"Unexpected Error: {str(e)}"
        with open(summary_file, 'a', encoding='utf-8') as sf:
            for calc in group:
                sf.write(f"{calc['base_name']};{calc['method']};{calc['form']};{job_id};{status}\n")
    finally:
        if script_path.exists():
            script_path.unlink()

def calculate_pka(xyz_dir, basis, methods, output_dir, forms=None, tasks_per_node=32):
    """Main function to calculate pKa values"""
    print(f"Starting pKa calculations for molecules in {xyz_dir}")
    print(f"Using basis set: {basis}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Output directory: {output_dir}")
    print(f"Tasks per node: {tasks_per_node}")
    
    generate_calculations(xyz_dir, basis, methods, output_dir, forms, tasks_per_node)
    
    print("Calculations submitted successfully!")