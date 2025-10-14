import re
import csv
from pathlib import Path
from datetime import datetime


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
    atoms = [line.strip().split() for line in lines[2:2 + num_atoms]]

    return num_atoms, comment, atoms


def create_equilibrated_xyz(trj_xyz_path, output_dir, molecule):
    """Create equilibrated XYZ file (last frame of trajectory, without modifications)"""
    num_atoms, comment, atoms = get_last_frame_from_trj(trj_xyz_path)

    new_content = f"{num_atoms}\n{comment}\n"
    new_content += "\n".join(" ".join(atom) for atom in atoms)

    output_path = output_dir / f"{molecule}.xyz"
    with open(output_path, 'w') as f:
        f.write(new_content)

    return output_path, num_atoms


def write_equilibrated_info_to_csv(csv_path, molecule, basis_set, method, timestamp, num_atoms):
    """Write information about equilibrated molecule to CSV file"""
    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'molecule', 'basis_set', 'method', 'num_atoms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp': timestamp,
            'molecule': molecule,
            'basis_set': basis_set,
            'method': method,
            'num_atoms': num_atoms
        })


def process_equilibrated(calc_dir, output_dir):
    """Process and save equilibrated XYZ molecules"""
    calc_dir = Path(calc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "equilibrated_molecules.csv"

    basis_sets = [d.name for d in calc_dir.iterdir() if d.is_dir()]
    if not basis_sets:
        print("No basis sets found in calculation directory")
        return

    print("\nAvailable basis sets:")
    for i, basis in enumerate(basis_sets, 1):
        print(f"{i}. {basis}")

    basis_choice = input("\nSelect basis set (number): ")
    try:
        selected_basis = basis_sets[int(basis_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    basis_path = calc_dir / selected_basis
    methods = set()
    molecules = set()

    for mol_dir in basis_path.iterdir():
        if mol_dir.is_dir():
            molecules.add(mol_dir.name)
            forms = set()
            for form_dir in mol_dir.iterdir():
                if form_dir.is_dir():
                    forms.add(form_dir.name)
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
        selected_method = sorted(methods)[int(method_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    print(f"\nProcessing equilibrated molecules with basis '{selected_basis}' and method '{selected_method}'...")
    print(f"Information will be saved to: {csv_path}")

    processed = 0
    for molecule in sorted(molecules):
        mol_dir = basis_path / molecule
        for form_dir in mol_dir.iterdir():
            if not form_dir.is_dir():
                continue
            for method_dir in form_dir.iterdir():
                if not method_dir.is_dir():
                    continue
    
                trj_path = method_dir / "input_trj.xyz"
                if not trj_path.exists():
                    print(f"Skipping {molecule}/{form_dir.name}/{method_dir.name}: input_trj.xyz not found")
                    continue
    
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    xyz_path, num_atoms = create_equilibrated_xyz(trj_path, output_dir, f"{molecule}{'' if form_dir.name == 'neutral' else '_' + form_dir.name}")
                    write_equilibrated_info_to_csv(
                        csv_path,
                        f"{molecule}_{form_dir.name}",
                        selected_basis,
                        method_dir.name,
                        timestamp,
                        num_atoms
                    )
                    processed += 1
                    print(f"Processed {molecule}/{form_dir.name} - saved equilibrated structure ({num_atoms} atoms)")
                except Exception as e:
                    print(f"Error processing {molecule}/{form_dir.name}: {e}")

    print(f"\nDone! Created {processed} equilibrated molecules in {output_dir}")
    print(f"Information saved to: {csv_path}")
