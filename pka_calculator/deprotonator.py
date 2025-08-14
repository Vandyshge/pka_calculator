import os
import re
import random
from pathlib import Path

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

def create_deprotonated_xyz(original_xyz_path, output_dir, hydrogen_id, molecule):
    """Create deprotonated XYZ file by removing specified hydrogen"""
    with open(original_xyz_path, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    atoms = [line.strip().split() for line in lines[2:2+num_atoms]]
    
    new_atoms = [atom for i, atom in enumerate(atoms) if i != hydrogen_id]
    
    new_content = f"{len(new_atoms)}\n{comment}\n"
    new_content += "\n".join(" ".join(atom) for atom in new_atoms)
    
    output_path = output_dir / f"{molecule}_deprotonated.xyz"
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    return output_path

def process_deprotonation(calc_dir, output_dir):
    """Interactive process to create deprotonated XYZ files"""
    calc_dir = Path(calc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available basis sets
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
    
    # Find available methods for selected basis
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
    
    processed = 0
    for molecule in sorted(molecules):
        output_path = basis_path / molecule / "neutral" / selected_method / "output.out"
        xyz_path = basis_path / molecule / "neutral" / selected_method / "molecule.xyz"
        
        if not output_path.exists() or not xyz_path.exists():
            continue
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        hydrogen_id = find_charged_hydrogen(content)
        if hydrogen_id is not None:
            create_deprotonated_xyz(xyz_path, output_dir, hydrogen_id, molecule)
            processed += 1
            print(f"Processed {molecule} - removed hydrogen {hydrogen_id}")
    
    print(f"\nDone! Created {processed} deprotonated molecules in {output_dir}")