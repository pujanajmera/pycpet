import numpy as np

from CPET.utils.calculator import (
    propagate_topo_dev,
    Inside_Box,
    compute_curv_and_dist
)

def task(x_0, n_iter, x, Q, step_size, dimensions):
    x_init = x_0
    for j in range(n_iter):
        # x_0 = propagate_topo_dev(x_0, self.x, self.Q, self.step_size)
        x_0 = propagate_topo_dev(x_0, x, Q, step_size)
        if not Inside_Box(x_0, dimensions):
            # count += 1
            break
    x_init_plus = propagate_topo_dev(x_init, x, Q, step_size)
    x_init_plus_plus = propagate_topo_dev(x_init_plus, x, Q, step_size)
    x_0_plus = propagate_topo_dev(x_0, x, Q, step_size)
    x_0_plus_plus = propagate_topo_dev(x_0_plus, x, Q, step_size)

    result = compute_curv_and_dist(
        x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
    )
    return result

