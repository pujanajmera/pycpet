import numpy as np
from CPET.source.calculator import calculator
import warnings
import time
import matplotlib.pyplot as plt
import json

warnings.filterwarnings(action="ignore")

from CPET.utils.calculator import (
    calculate_field_at_point,
    calculate_electric_field_base,
    calculate_electric_field_c_shared_full_alt,
)


def test_specific_field(x_0, x, Q, field_func):
    """
    Test the electric field calculation using a specific electric field function
    """
    start = time.time()
    field = field_func(x_0, x, Q)
    end = time.time()
    dt = end - start
    return dt, field


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
    function_list = [
        calculate_electric_field_base,
        # calculate_electric_field_dev_c_shared,
        calculate_electric_field_c_shared_full_alt,
        # calculate_electric_field_gpu_for_test,
        # calculate_field_at_point,
    ]
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
        x_0 = calc.center
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
            # Test the electric field function, in triplicate
            try:
                dt1, field = test_specific_field(x_0, x, Q, j)
                dt2, field = test_specific_field(x_0, x, Q, j)
                dt3, field = test_specific_field(x_0, x, Q, j)
                dt = np.mean([dt1, dt2, dt3])
                dt_std = np.std([dt1, dt2, dt3])
                print(f"Field computed: {field}")
                print(
                    f"Time taken for {j.__name__} on {i} system: {dt:.4f}+-{dt_std:.4f} seconds"
                )
            except Exception as e:
                print(f"Error in {j.__name__} on {i} system: {e}")
            # Store the results of time, only related to quantifiable parameters
            results.append(
                [j.__name__, i, dt, dt_std, n_charges, dimensions, step_size]
            )

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
