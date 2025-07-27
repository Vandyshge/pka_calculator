import os
import subprocess
from pathlib import Path

METHOD_TEMPLATES = {
    'HF': "! HF {basis} TightSCF CPCM(water) OPT Freq",
    'B3LYP': "! B3LYP {basis} TightSCF CPCM(water) OPT Freq",
    'PBE0': "! PBE0 {basis} TightSCF CPCM(water) OPT Freq",
    'PBE': "! PBE {basis} TightSCF CPCM(water) OPT Freq"
}

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

def generate_calculations(xyz_dir, basis, methods, output_dir):
    xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith(".xyz") and not f.endswith("_deprotonated.xyz")]
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / "calculations_summary.csv"
    
    with open(summary_file, 'w', encoding='utf-8') as sf:
        sf.write("Molecule;Method;Form;Job ID;Status\n")
    
    for xyz in xyz_files:
        base_name = os.path.splitext(xyz)[0]
        
        for form, suffix in [("neutral", ""), ("deprotonated", "_deprotonated")]:
            src_xyz = Path(xyz_dir) / f"{base_name}{suffix}.xyz"
            if not src_xyz.exists():
                continue
                
            for method in methods:
                if method not in METHOD_TEMPLATES:
                    continue
                    
                method_name = method
                method_dir = output_dir / basis / base_name / form / method_name
                method_dir.mkdir(parents=True, exist_ok=True)

                charge = 0 if form == "neutral" else -1
                multiplicity = calculate_multiplicity(src_xyz, charge=charge)
                
                # Write input file
                input_file = method_dir / "input.inp"
                with open(input_file, 'w', encoding='utf-8') as f:
                    f.write(METHOD_TEMPLATES[method].format(basis=basis) + "\n")
                    f.write(f'''%geom  
   MaxIter 200
end''')
                    f.write(f"* xyzfile {charge} {multiplicity} molecule.xyz\n")
             
                # Copy XYZ file
                os.system(f"cp {src_xyz} {method_dir}/molecule.xyz")
                
                # Submit job with sbatch
                os.chdir(method_dir)
                try:
                    sbatch_cmd = [
                        'sbatch',
                        '--nodes=1',
                        '--ntasks=1',
                        '--job-name=orca',
                        '--wrap', f"orca input.inp > output.out"
                    ]
                    
                    result = subprocess.run(
                        sbatch_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    
                    job_id = result.stdout.strip().split()[-1]
                    status = "Submitted"
                    
                except subprocess.CalledProcessError as e:
                    job_id = "Failed"
                    status = f"Slurm Error: {e.stderr.strip()}"
                except Exception as e:
                    job_id = "Failed"
                    status = f"Unexpected Error: {str(e)}"

                with open(summary_file, 'a', encoding='utf-8') as sf:
                    sf.write(f"{base_name};{method_name};{form};{job_id};{status}\n")
                
                os.chdir("../../../../../")  # Return to original directory

def calculate_pka(xyz_dir, basis, methods, output_dir):
    """Main function to calculate pKa values"""
    print(f"Starting pKa calculations for molecules in {xyz_dir}")
    print(f"Using basis set: {basis}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Output directory: {output_dir}")
    
    generate_calculations(xyz_dir, basis, methods, output_dir)
    
    print("Calculations submitted successfully!")