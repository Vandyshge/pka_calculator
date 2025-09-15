import os
import re
import random
import csv
from pathlib import Path
from datetime import datetime

def find_charged_hydrogen(output_content):
    """Find the hydrogen with highest charge in Mulliken analysis"""
    sections = re.findall(
        r'-+\nMULLIKEN ATOMIC CHARGES\n-+\n(.*?)\nSum of atomic charges:.*?\n',
        output_content,
        re.DOTALL
    )
    
    if not sections:
        return None
    
    last_section = sections[-1]
    hydrogens = []
    
    for line in last_section.split('\n'):
        if ' H :' in line:
            parts = line.split()
            atom_id = int(parts[0])
            charge = float(parts[3])
            hydrogens.append((atom_id, charge))
    
    if not hydrogens:
        return None
    
    max_charge = max(h[1] for h in hydrogens)
    candidates = [h[0] for h in hydrogens if h[1] == max_charge]
    return random.choice(candidates)

def get_last_frame_from_trj(trj_file_path):
    """Extract the last frame from input_trj.xyz file"""
    with open(trj_file_path, 'r') as f:
        content = f.read()
    
    frames = re.findall(r'(\d+\n.*?)(?=\n\d+\n|\Z)', content, re.DOTALL)
    
    if not frames:
        raise ValueError("No frames found in input_trj.xyz")
    
    last_frame = frames[-1].strip()
    lines = last_frame.split('\n')
    
    num_atoms = int(lines[0])
    comment = lines[1]
    atoms = [line.strip().split() for line in lines[2:2+num_atoms]]
    
    return num_atoms, comment, atoms

def create_deprotonated_xyz(trj_xyz_path, output_dir, hydrogen_id, molecule):
    """Create deprotonated XYZ file by removing specified hydrogen from last frame"""
    num_atoms, comment, atoms = get_last_frame_from_trj(trj_xyz_path)
    
    hydrogen_info = atoms[hydrogen_id] if hydrogen_id < len(atoms) else None
    
    new_atoms = [atom for i, atom in enumerate(atoms) if i != hydrogen_id]
    
    new_content = f"{len(new_atoms)}\n{comment}\n"
    new_content += "\n".join(" ".join(atom) for atom in new_atoms)
    
    output_path = output_dir / f"{molecule}_deprotonated.xyz"
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    return output_path, hydrogen_info

def write_removal_info_to_csv(csv_path, molecule, basis_set, method, hydrogen_id, hydrogen_info, timestamp):
    """Write information about removed hydrogen to CSV file"""
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'molecule', 'basis_set', 'method', 'hydrogen_id', 
                     'element', 'x_coord', 'y_coord', 'z_coord']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        if hydrogen_info and len(hydrogen_info) >= 4:
            writer.writerow({
                'timestamp': timestamp,
                'molecule': molecule,
                'basis_set': basis_set,
                'method': method,
                'hydrogen_id': hydrogen_id,
                'element': hydrogen_info[0],
                'x_coord': hydrogen_info[1],
                'y_coord': hydrogen_info[2],
                'z_coord': hydrogen_info[3]
            })
        else:
            writer.writerow({
                'timestamp': timestamp,
                'molecule': molecule,
                'basis_set': basis_set,
                'method': method,
                'hydrogen_id': hydrogen_id,
                'element': 'Unknown',
                'x_coord': 'N/A',
                'y_coord': 'N/A',
                'z_coord': 'N/A'
            })

def process_deprotonation(calc_dir, output_dir):
    """Interactive process to create deprotonated XYZ files"""
    calc_dir = Path(calc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "removed_hydrogens.csv"
    
    basis_sets = [d.name for d in calc_dir.iterdir() if d.is_dir()]
    if not basis_sets:
        print("No basis sets found in calculation directory")
        return
    
    print("\nAvailable basis sets:")
    for i, basis in enumerate(basis_sets, 1):
        print(f"{i}. {basis}")
    
    basis_choice = input("\nSelect basis set (number): ")
    try:
        selected_basis = basis_sets[int(basis_choice)-1]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    basis_path = calc_dir / selected_basis
    methods = set()
    molecules = set()
    
    for mol_dir in basis_path.iterdir():
        if mol_dir.is_dir():
            molecules.add(mol_dir.name)
            for form_dir in mol_dir.iterdir():
                if form_dir.is_dir() and form_dir.name == "neutral":
                    for method_dir in form_dir.iterdir():
                        if method_dir.is_dir():
                            methods.add(method_dir.name)
    
    if not methods:
        print("No methods found for selected basis set")
        return
    
    print("\nAvailable methods:")
    for i, method in enumerate(sorted(methods), 1):
        print(f"{i}. {method}")
    
    method_choice = input("\nSelect method (number): ")
    try:
        selected_method = sorted(methods)[int(method_choice)-1]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    print(f"\nProcessing molecules with basis '{selected_basis}' and method '{selected_method}'...")
    print(f"Removal information will be saved to: {csv_path}")
    
    processed = 0
    for molecule in sorted(molecules):
        output_path = basis_path / molecule / "neutral" / selected_method / "output.out"
        trj_path = basis_path / molecule / "neutral" / selected_method / "input_trj.xyz"
        
        if not output_path.exists():
            print(f"Skipping {molecule}: output.out not found")
            continue
        if not trj_path.exists():
            print(f"Skipping {molecule}: input_trj.xyz not found")
            continue
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        hydrogen_id = find_charged_hydrogen(content)
        if hydrogen_id is not None:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                xyz_path, hydrogen_info = create_deprotonated_xyz(trj_path, output_dir, hydrogen_id, molecule)
                write_removal_info_to_csv(csv_path, molecule, selected_basis, selected_method, 
                                         hydrogen_id, hydrogen_info, timestamp)
                processed += 1
                print(f"Processed {molecule} - removed hydrogen {hydrogen_id} (coordinates: {hydrogen_info[1:] if hydrogen_info else 'N/A'})")
            except Exception as e:
                print(f"Error processing {molecule}: {e}")
        else:
            print(f"No charged hydrogen found in {molecule}")
    
    print(f"\nDone! Created {processed} deprotonated molecules in {output_dir}")
    print(f"Removal information saved to: {csv_path}")