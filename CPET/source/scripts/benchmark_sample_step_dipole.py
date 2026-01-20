import os
import numpy as np
from glob import glob
from random import choice
from CPET.source.calculator import calculator
from CPET.utils.calculator import make_histograms, construct_distance_matrix, initialize_box_points_uniform
from CPET.source.benchmark import gen_param_dist_mat
import argparse
import matplotlib.pyplot as plt
from CPET.source.CPET import CPET
import logging

def main():
    parser = argparse.ArgumentParser(
        description="Compute electric field topology from a set of input dipoles"
    )
    parser.add_argument(
        "--dipole_dir",
        "-d",
        type=str,
        required=True,
        help="Path to the dipole file dir. Each filename needs to be unique with the following columns: index, x(A), y(A), z(A), mx(a.u.), my(a.u.), mz(a.u.)",
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
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads to use for computation (default: 1)",
    )
    args = parser.parse_args()

    center = np.array(args.center)
    x = np.array(args.x)
    y = np.array(args.y)
    print("Center:", center)
    print("X-axis:", x)
    print("Y-axis:", y)

    dimensions = np.array(args.dimensions)
    print("Box dimensions:", dimensions)
    files_input = glob(args.dipole_dir + f"/*.dip")

    num = 3
    if len(files_input) != 3:
        raise ValueError("Please provide exactly 3 dipole files for benchmarking")

    topo_files = []
    benchmark_step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    benchmark_samples = [500000, 100000, 50000, 10000, 5000, 1000]
    for step_size in benchmark_step_sizes:
        for n_samples in benchmark_samples:
            for i in range(3):
                for file in files_input:
                    files_done = [
                        x
                        for x in os.listdir("./")
                        if x.split(".")[-1] == "top"
                    ]
                    dipoles = np.loadtxt(file, skiprows=1, dtype=np.float32)
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
                    dipole_positions = (dipole_positions - center) @ np.linalg.inv(
                        transformation_matrix
                    )
                    dipole_moments = dipoles[:, 4:7]  # Already in atomic units
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
                    topo_calc_dip.center = center 
                    logging.basicConfig(level=logging.DEBUG)
                    topo_calc_dip.log = logging.getLogger(__name__) 
                    # Convert to float32
                    topo_calc_dip.x = topo_calc_dip.x.astype(np.float32)
                    topo_calc_dip.mu = topo_calc_dip.mu.astype(np.float32)
                
                    hist = topo_calc_dip.compute_topo_complete_c_shared_dipole()
                
                    # Save hist based on the dipole file name in working directory
                    dipole = file.split("/")[-1].split(".")[0]
                    outstring = "{}_{}_{}_{}.top".format(
                        dipole, n_samples, str(step_size)[2:], i
                    )
                    if outstring in files_done:
                        topo_files.append("./" + outstring)
                        continue
                    np.savetxt("./" + outstring, hist)
                    topo_files.append("./" + outstring)
    for file in files_input:
        topo_file_protein = [
            x for x in topo_files if file.split("/")[-1].split(".")[0] in x
        ]
        if len(topo_file_protein) != num * len(benchmark_samples) * len(
            benchmark_step_sizes
        ):
            raise ValueError(
                "Incorrect number of output topologies for requested benchmark parameters"
            )
        histograms = make_histograms(topo_file_protein, plot=False)
        plt.close()
        distance_matrix = construct_distance_matrix(histograms)
        avg_dist = gen_param_dist_mat(distance_matrix, topo_file_protein)


if __name__ == "__main__":
    main()

