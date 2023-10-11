import numpy as np

from CPET.utils.calculator import (
    propagate_topo,
    Inside_Box,
)


def wrapper_single_job(args):
    n_iter, x_0, x, x_init, Q, step_size, center, dimensions = args
    count = 0

    for j in range(n_iter):
        x_0 = propagate_topo(x_0, x, Q, step_size)
        if np.linalg.norm(x_0[0] - np.array(center)) >= 1.5:
            if not Inside_Box(x_0[0], dimensions):
                count += 1
                break

    x_init_plus = propagate_topo(x_init, x, Q, step_size)
    x_init_plus_plus = propagate_topo(x_init_plus, x, Q, step_size)
    x_0_plus = propagate_topo(x_0, x, Q, step_size)
    x_0_plus_plus = propagate_topo(x_0_plus, x, Q, step_size)

    # Parallel # Parallel # Parallel # Parallel # Parallel# Parallel # Parallel
    # Set the start method to 'spawn'
    """args_list = [
        (x_init, x, Q, options["step_size"]),
        (x_0, x, Q, options["step_size"]),
    ]
    with multiprocessing.Pool(processes=2) as pool:
        # Use the pool to parallelize the function calls
        results = pool.map(propagate_topo_wrapper, args_list)

    x_init_plus, x_init_plus_plus = results[0]
    x_0_plus, x_0_plus_plus = results[1]
    """
    # Parallel # Parallel # Parallel # Parallel # Parallel# Parallel # Parallel

    return (
        x_init,
        x_init_plus,
        x_init_plus_plus,
        x_0,
        x_0_plus,
        x_0_plus_plus,
    )


def propagate_topo_wrapper(args):
    x_init, x, Q, step_size = args
    x_init_plus = propagate_topo(x_init, x, Q, step_size)
    return x_init_plus, propagate_topo(x_init_plus, x, Q, step_size)
