import numpy as np
import argparse
from CPET.source.calculator import calculator
from CPET.utils.calculator import initialize_box_points_uniform

def main():
    parser = argparse.ArgumentParser(
        description="Compute electric field topology from a set of input dipoles"
    )
    parser.add_argument(
        "--dipole-file",
        "-d",
        type=str,
        required=True,
        help="Path to the dipole file. Needs to have a header and a fairly unique name, and then the following columns: index, x(A), y(A), z(A), mx(a.u.), my(a.u.), mz(a.u.)",
    )
    parser.add_argument(
        "--center",
        "-c",
        type=float,
        default=[0.0, 0.0, 0.0],
        help="List of 3 coordinates to define the center",
        nargs=3,
    )
    parser.add_argument(
        "-x",
        type=float,
        default=[1.0, 0.0, 0.0],
        help="List of 3 coordinates to define the x-axis relative to center",
        nargs=3,
    )
    parser.add_argument(
        "-y",
        type=float,
        default=[1.0, 0.0, 0.0],
        help="List of 3 coordinates to define the y-axis relative to center",
        nargs=3,
    )
    parser.add_argument(
        "--dimensions",
        "-dim",
        type=np.float32,
        default=[1.0, 1.0, 1.0],
        help="List of 3 coordinates to define the box dimensions. These correspond to half-widths of the box",
        nargs=3,
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=1000,
        help="Number of streamlines to compute the electric field topology (default: 1000)",
    )
    parser.add_argument(
        "--step-size",
        "-s",
        type=float,
        default=0.1,
        help="Step size for the propagation of the streamlines (default: 0.1)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads to use for computation (default: 1)",
    )
    args = parser.parse_args()

    n_samples = args.n_samples
    step_size = args.step_size

    center = np.array(args.center)
    x = np.array(args.x)
    y = np.array(args.y)
    print("Center:", center)
    print("X-axis:", x)
    print("Y-axis:", y)

    dimensions = np.array(args.dimensions)
    print("Box dimensions:", dimensions)

    dipoles = np.loadtxt(args.dipole_file, skiprows=1, dtype=np.float32)
    if dipoles.shape[1] != 7:
        raise ValueError(
            "Dipole file must have 7 columns: index, x(A), y(A), z(A), mx(a.u.), my(a.u.), mz(a.u.)"
        )

    max_steps = round(2 * np.linalg.norm(dimensions) / step_size)
    num_per_dim = round(n_samples ** (1 / 3))
    if num_per_dim**3 < n_samples:
        num_per_dim += 1
    n_samples = num_per_dim**3
    seed = None
    grid_density = 2 * dimensions / (num_per_dim + 1)
    print("grid_density: ", grid_density)

    (
        random_start_points,
        random_max_samples,
        transformation_matrix,
    ) = initialize_box_points_uniform(
        center=center,
        x=x,
        y=y,
        dimensions=dimensions,
        N_cr=[num_per_dim, num_per_dim, num_per_dim],
        dtype="float32",
        max_steps=max_steps,
        ret_rand_max=True,
        inclusive=False,
        seed=seed,
    )
    random_start_points = random_start_points.reshape(-1, 3)
    n_samples = len(random_start_points)

    dipole_positions = dipoles[:, 1:4]
    dipole_positions = (dipole_positions - center) @ np.linalg.inv(transformation_matrix)
    dipole_moments = dipoles[:, 4:7] # Already in atomic units
    dipole_moments = dipole_moments @ np.linalg.inv(transformation_matrix)

    # Create an instance without __init__
    topo_calc_dip = calculator.__new__(calculator)
    topo_calc_dip.n_samples = n_samples
    topo_calc_dip.step_size = step_size
    topo_calc_dip.x = dipole_positions
    topo_calc_dip.mu = dipole_moments
    topo_calc_dip.dimensions = dimensions
    topo_calc_dip.transformation_matrix = transformation_matrix
    topo_calc_dip.random_start_points = random_start_points
    topo_calc_dip.random_max_samples = random_max_samples
    topo_calc_dip.concur_slip = args.threads

    #Convert to float32
    topo_calc_dip.x = topo_calc_dip.x.astype(np.float32)
    topo_calc_dip.mu = topo_calc_dip.mu.astype(np.float32)

    hist = topo_calc_dip.compute_topo_complete_c_shared_dipole()

    # Save hist based on the dipole file name in working directory
    filename = args.dipole_file.split("/")[-1].split(".")[0]
    np.savetxt("{}.top".format(filename), hist)

if __name__ == "__main__":
    main()
