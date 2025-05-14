import numpy as np
from CPET.source.calculator import calculator
import warnings
import time
import matplotlib.pyplot as plt
import json

warnings.filterwarnings(action="ignore")
import torch
import pkg_resources

from CPET.utils.calculator import (
    calculate_field_at_point,
    calculate_electric_field_base,
    calculate_electric_field_c_shared_full_alt,
)

from CPET.utils.gpu import calculate_electric_field_torch_batch_gpu

from CPET.utils.c_ops import Math_ops

package_name = "pycpet"
package = pkg_resources.get_distribution(package_name)
package_path = package.location
Math = Math_ops(shared_loc=package_path + "/CPET/utils/math_module.so")


def loop_field_simult_c_shared_full(x_0, x, Q):
    """
    Computes the electric field from x_0, x, Q, at all points simultaneously
    Batched
    Takes:
        x_0(array) - (L,3) array of points to compute field at
        x(np array) - (N,3) array of points to compute field from
        Q(np array) - (N,1) array of charge values
    Returns:
        field(array) - (L, 1) array of electric field
    """
    field = Math.compute_looped_field(x_0, x, Q)
    return field


def grid_field_simult_c_shared_full(x_0, x, Q):
    """
    Computes the electric field from x_0, x, Q, at all points simultaneously
    Batched
    Takes:
        x_0(array) - (L,3) array of points to compute field at
        x(np array) - (N,3) array of points to compute field from
        Q(np array) - (N,1) array of charge values
    Returns:
        field(array) - (L, 1) array of electric field
    """
    field = Math.compute_batch_field(x_0, x, Q, int(100))
    return field


def test_specific_field_grid_simult(x_0, x, Q, field_func):
    """
    Test the electric field calculation using a specific electric field function
    """
    start = time.time()
    field = field_func(x_0, x, Q)
    end = time.time()
    dt = end - start
    return dt, field


def test_specific_field_grid_loop(x_0, x, Q, field_func):
    """
    Test the electric field calculation using a specific electric field function
    """
    start = time.time()
    total_field = np.zeros_like(x_0)
    for i in range(len(x_0)):
        total_field[i] = field_func(x_0[i], x, Q)
    end = time.time()
    dt = end - start
    return dt, total_field


def main():
    """
    Loop over several electric field functions and test them, the following functions/parameters are tested:

    - Individual Fields:
        - calculate_electric_field_dev_c_shared
        - calculate_electric_field_base
        - calculate_electric_field_dev_c_shared
        - calculate_electric_field_c_shared_full_alt
        - calculate_electric_field_c_shared_full
        - calculate_electric_field_gpu_for_test
    - Grid Fields:
        - compute_field_on_grid
    - System Sizes (Varying the size of x):
        - "tiny": ~10^3 points
        - "small": ~10^4 points
        - "medium": ~10^5 points
        - "large": ~10^6 points
    """

    plot = True

    radius_filter_dict = {"tiny": 12, "small": 30, "medium": None, "large": None}

    results = []
    function_list_single = [
        calculate_electric_field_base,
        calculate_electric_field_c_shared_full_alt,
        # calculate_field_at_point,
    ]
    function_list_grid = [
        calculate_electric_field_torch_batch_gpu,
        loop_field_simult_c_shared_full,
        grid_field_simult_c_shared_full,
    ]

    function_list = function_list_single + function_list_grid

    for i in [
        "tiny",
        "small",
        "medium",
    ]:  # Skipping large for now, not benchmarking on systems this large
        "Running tests for {} system".format(i)
        # Load the PDB file
        pdb_file = "./test_files/test.pdb"
        # Load the options file from the test_files directory
        options = json.load(open("./test_files/options_test.json"))
        if radius_filter_dict[i] is not None:
            options["filter_radius"] = radius_filter_dict[i]
        calc = calculator(options, pdb_file)
        x_0 = calc.mesh.reshape(-1, 3)
        x = calc.x
        Q = calc.Q
        dimensions = calc.dimensions
        step_size = calc.step_size

        n_charges = len(Q)
        print(
            f"Number of charges: {n_charges} by radius filter {radius_filter_dict[i]}"
        )

        # Test the electric field functions, shuffle their order to avoid bias in timing
        function_list_shuffled = np.random.permutation(function_list)
        for j in function_list_shuffled:
            print(j)
            # Test the electric field function, in triplicate. If error, show full error including line number
            if j in function_list_single:
                try:
                    x_0 = calc.mesh.reshape(-1, 3)
                    x = calc.x
                    Q = calc.Q
                    dt1, field = test_specific_field_grid_loop(x_0, x, Q, j)
                    dt2, field = test_specific_field_grid_loop(x_0, x, Q, j)
                    dt3, field = test_specific_field_grid_loop(x_0, x, Q, j)
                    dt = np.mean([dt1, dt2, dt3])
                    dt_std = np.std([dt1, dt2, dt3])
                    print(f"Field computed: {field}")
                    print(
                        f"Time taken for {j} on {i} system: {dt:.4f}+-{dt_std:.4f} seconds"
                    )
                except Exception as e:
                    print(f"Error in {j} on {i} system: {e}")
                    # Print the traceback to see where the error occurred
                    import traceback

                    traceback.print_exc()
                    exit()
            elif j in function_list_grid:
                # Test the electric field function, in triplicate
                try:
                    if j is calculate_electric_field_torch_batch_gpu:
                        print("Converting arrays to tensors")
                        x_0 = torch.tensor(x_0, dtype=torch.float32).cuda()
                        x = torch.tensor(x, dtype=torch.float32).cuda()
                        Q = torch.tensor(Q, dtype=torch.float32).cuda()
                    else:
                        x_0 = calc.mesh.reshape(-1, 3)
                        x = calc.x
                        Q = calc.Q
                    dt1, field = test_specific_field_grid_simult(x_0, x, Q, j)
                    dt2, field = test_specific_field_grid_simult(x_0, x, Q, j)
                    dt3, field = test_specific_field_grid_simult(x_0, x, Q, j)
                    dt = np.mean([dt1, dt2, dt3])
                    dt_std = np.std([dt1, dt2, dt3])
                    print(f"Field computed: {field}")
                    print(
                        f"Time taken for {j} on {i} system: {dt:.4f}+-{dt_std:.4f} seconds"
                    )
                except Exception as e:
                    print(f"Error in {j} on {i} system: {e}")
                    # Print the traceback to see where the error occurred
                    import traceback

                    traceback.print_exc()
                    exit()
            # Store the results of time, only related to quantifiable parameters
            results.append([j, i, dt, dt_std, n_charges, dimensions, step_size])

    # Only setting up plotting for n_charges vs time across different field functions
    if plot:
        # Plot all time vs n_charges for each field function, overlayed
        plt.figure(figsize=(10, 6))
        print(results)
        for func in set([x[0] for x in results]):
            times = []
            times_std = []
            n_charges = []
            for result in results:
                if result[0] == func:
                    times.append(result[2])
                    times_std.append(result[3])
                    n_charges.append(result[4])

            color = next(plt.gca()._get_lines.prop_cycler)["color"]
            # Plotting errorbars and points simultaneously
            plt.plot(n_charges, times, "-o", label=func, color=color)
            plt.errorbar(
                n_charges, times, yerr=times_std, fmt="-o", capsize=5, color=color
            )
        # plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Number of Charges (n_charges)")
        plt.ylabel("Time (seconds)")
        plt.title("Performance Comparison of Electric Field Calculator Functions")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
