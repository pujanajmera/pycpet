import json
import numpy as np

def parse_pdb(pdb_file_path):
    atom_info = []
    with open(pdb_file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                atom_number = int(line[6:11].strip())
                atom_type = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_number = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_info.append((atom_number, atom_type, residue_name, residue_number, x, y, z))
    return atom_info

def calculate_coordinate(atoms, method):
    coordinates = np.array([atom[-3:] for atom in atoms])
    if method == "mean":
        return np.mean(coordinates, axis=0)
    elif method == "first":
        return coordinates[0]
    elif method == "inverse":
        first_atom = coordinates[0]
        average_of_others = np.mean(coordinates[1:], axis=0)
        return 2 * average_of_others - first_atom

def parse_options_file(options_file_path):
    with open(options_file_path, 'r') as file:
        options = json.load(file)
    return options

def main(options_file_path):
    options = parse_options_file(options_file_path)
    atom_data = parse_pdb(options["pdb"])
    
    final_values = {}
    for key in ["center", "x", "y"]:
        method = options[key]["method"]
        input_atoms = [(atom_type, residue_number) for atom_type, residue_number in options[key].items() if atom_type != "method"]
        atoms_to_consider = [atom for atom in atom_data if (atom[1], atom[3]) in input_atoms]
        
        final_values[key] = calculate_coordinate(atoms_to_consider, method)
    
    print("Box Center:", final_values["center"])
    print("Box X:", final_values["x"])
    print("Box Y:", final_values["y"])

# Example usage
if __name__ == "__main__":
    options_file_path = 'options_atom_sel.json'
    main(options_file_path)
