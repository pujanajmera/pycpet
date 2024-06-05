import numpy as np
import time
import json

from CPET.utils.calculator import (
    calculate_electric_field,
    curv,
    compute_curv_and_dist,
    Inside_Box,
)
from CPET.utils.parser import parse_pdb, calculate_center, initialize_box_points


def propagate_topo(x_0, x, Q, step_size):
    """
    Propagates position based on normalized electric field at a given point
    Takes
        x_0(array) - position to propagate based on field at that point of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
        step_size(float) - size of streamline step to take when propagating, real and positive
    Returns
        x_0 - new position on streamline after propagation via electric field
    """
    E = calculate_electric_field(x_0, x, Q)  # Compute field
    E = E / np.linalg.norm(E)
    x_0 = x_0 + step_size * E
    return x_0


def main():
    """
    default_options = {
        "path_to_pdb": "./1_wt_run1_0.pqr",
        "center": [55.965, 46.219, 22.123],
        "x": [56.191, 48.344, 22.221],
        "y": [57.118, 46.793, 20.46],
        "n_samples": 100,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.001,
    }
    """

    options_file_path = "options_atom_sel.json"

    with open(options_file_path, "r") as file:
        options = json.load(file)

    atom_data = parse_pdb(options["pdb"])
    x, Q = parse_pdb(
        options["path_to_pdb"],
    )

    final_values = {}
    for key in ["center", "x", "y"]:
        method = options[key]["method"]
        input_atoms = [
            (atom_type, residue_number)
            for atom_type, residue_number in options[key].items()
            if atom_type != "method"
        ]
        atoms_to_consider = [
            atom for atom in atom_data if (atom[1], atom[3]) in input_atoms
        ]
        final_values[key] = calculate_center(atoms_to_consider, method)

    center = np.array(options["center"])
    x_vec_pt = np.array(options["x"])
    y_vec_pt = np.array(options["y"])
    dimensions = np.array(options["dimensions"])
    step_size = options["step_size"]
    n_samples = options["n_samples"]

    (
        random_start_points,
        random_max_samples,
        transformation_matrix,
    ) = initialize_box_points(
        center, x_vec_pt, y_vec_pt, dimensions, n_samples, step_size
    )
    hist = []
    start_time = time.time()
    count = 0
    x = (x - center) @ np.linalg.inv(transformation_matrix)
    for idx, i in enumerate(random_start_points):
        x_0 = i
        x_init = x_0
        n_iter = random_max_samples[idx]
        for j in range(n_iter):
            x_0 = propagate_topo(x_0, x, Q, step_size)
            if not Inside_Box(x_0, dimensions):
                count += 1
                break
        x_init_plus = propagate_topo(x_init, x, Q, step_size)
        x_init_plus_plus = propagate_topo(x_init_plus, x, Q, step_size)
        x_0_plus = propagate_topo(x_0, x, Q, step_size)
        x_0_plus_plus = propagate_topo(x_0_plus, x, Q, step_size)
        hist.append(
            compute_curv_and_dist(
                x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
            )
        )
    end_time = time.time()
    np.savetxt("hist_cpet.txt", hist)
    print(
        f"Time taken for {options['n_samples']} calculations with N~{len(Q)}: {end_time - start_time:.2f} seconds"
    )
    print(count, len(random_start_points))


main()
