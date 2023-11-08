import numpy as np
import time
from multiprocessing import Pool

from CPET.utils.calculator import (
    calculate_electric_field_dev_c_shared,
    calculate_electric_field,
    compute_curv_and_dist,
    Inside_Box,
)
from CPET.utils.parser import parse_pqr
from CPET.utils.c_ops import Math_ops
from CPET.utils.fastmath import nb_subtract, power, nb_norm, nb_cross


"""class calculator:
    def __init__(self, math_loc="../utils/math_module.so"):
        self.math = Math_ops(shared_loc=math_loc)

    def __call__(self, x_0, x, Q):

        # Create matrix R
        R = nb_subtract(x_0, x)
        R_sq = R**2
        r_mag_sq = self.math.einsum_ij_i(R_sq).reshape(-1, 1)
        r_mag_cube = np.power(r_mag_sq, 3 / 2)
        E = self.math.einsum_operation(R, 1 / r_mag_cube, Q)

        return E
"""


math = Math_ops(shared_loc="../utils/math_module.so")


class Topo_calc:
    def __init__(self, options, math_loc="../utils/math_module.so"):
        self.efield_calc = calculator(math_loc=math_loc)
        self.options = options

        self.path_to_pqr = options["path_to_pqr"]
        self.center = np.array(options["center"])
        self.x_vec_pt = np.array(options["x"])
        self.y_vec_pt = np.array(options["y"])
        self.dimensions = np.array(options["dimensions"])
        self.step_size = options["step_size"]
        self.n_samples = options["n_samples"]
        self.concur_slip = options["concur_slip"]
        self.x, self.Q = parse_pqr(self.path_to_pqr)

        (
            self.random_start_points,
            self.random_max_samples,
            self.transformation_matrix,
        ) = initialize_box_points(
            self.center,
            self.x_vec_pt,
            self.y_vec_pt,
            self.dimensions,
            self.n_samples,
            self.step_size,
        )
        self.x = (self.x - self.center) @ np.linalg.inv(self.transformation_matrix)
        print("... > Initialized Topo_calc!")

    """def propagate_topo_dev(self, x_0):
        # if math is None:
        # Compute field
        # E = calculate_electric_field_dev_c_shared(x_0, self.x, self.Q, Math=math)
        E = calculate_electric_field(x_0, self.x, self.Q)
        # E = self.efield_calc(x_0, self.x, self.Q)
        # E = calculate_electric_field_cupy(x_0, x, Q)
        E = E / np.linalg.norm(E)
        x_0 = x_0 + self.step_size * E
        return x_0"""

    """def task(self, x_0, n_iter):
        x_init = x_0
        for j in range(n_iter):
            # x_0 = propagate_topo_dev(x_0, self.x, self.Q, self.step_size)
            x_0 = self.propagate_topo_dev(x_0, self.x, self.Q, self.step_size)
            if not Inside_Box(x_0, self.dimensions):
                # count += 1
                break
        x_init_plus = self.propagate_topo_dev(x_init, self.x, self.Q, self.step_size)
        x_init_plus_plus = self.propagate_topo_dev(
            x_init_plus, self.x, self.Q, self.step_size
        )
        x_0_plus = self.propagate_topo_dev(x_0, self.x, self.Q, self.step_size)
        x_0_plus_plus = self.propagate_topo_dev(
            x_0_plus, self.x, self.Q, self.step_size
        )

        result = compute_curv_and_dist(
            x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
        )
        return result"""

    def compute_topo(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            hist = pool.starmap(task, args)
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist


def initialize_box_points(center, x, y, dimensions, n_samples, step_size):
    """
    Initializes random points in box centered at the origin
    Takes
        center(array) - center of box of shape (1,3)
        x(array) - point to create x axis from center of shape (1,3)
        y(array) - point to create x axis from center of shape (1,3)
        dimensions(array) - L, W, H of box of shape (1,3)
        n_samples(int) - number of samples to compute
        step_size(float) - step_size of box
    Returns
        random_points_local(array) - array of random starting points in the box of shape (n_samples,3)
        random_max_samples(array) - array of maximum sample number for each streamline of shape (n_samples, 1)
        transformation_matrix(array) - matrix that contains the basis vectors for the box of shape (3,3)
    """
    # Convert lists to numpy arrays
    x = x - center  # Translate to origin
    y = y - center  # Translate to origin
    half_length, half_width, half_height = dimensions
    # Normalize the vectors
    x_unit = x / np.linalg.norm(x)
    y_unit = y / np.linalg.norm(y)
    # Calculate the z unit vector by taking the cross product of x and y
    z_unit = np.cross(x_unit, y_unit)
    z_unit = z_unit / np.linalg.norm(z_unit)
    # Recalculate the y unit vector
    y_unit = np.cross(z_unit, x_unit)
    y_unit = y_unit / np.linalg.norm(y_unit)
    # Generate random samples in the local coordinate system of the box
    random_x = np.random.uniform(-half_length, half_length, n_samples)
    random_y = np.random.uniform(-half_width, half_width, n_samples)
    random_z = np.random.uniform(-half_height, half_height, n_samples)
    # Each row in random_points_local corresponds to x, y, and z coordinates of a point in the box's coordinate system
    random_points_local = np.column_stack([random_x, random_y, random_z])
    # Convert these points back to the global coordinate system
    transformation_matrix = np.column_stack(
        [x_unit, y_unit, z_unit]
    ).T  # Each column is a unit vector
    max_distance = 2 * np.linalg.norm(
        np.array(dimensions)
    )  # Define maximum sample limit as 2 times the diagonal
    random_max_samples = np.random.randint(1, max_distance / step_size, n_samples)
    return random_points_local, random_max_samples, transformation_matrix


def propagate_topo_dev(x_0, x, Q, step_size):
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
    # if math is None:
    # Compute field
    E = calculate_electric_field_dev_c_shared(x_0, x, Q, Math=math)
    # E = calculate_electric_field(x_0, self.x, self.Q)
    #E = self.efield_calc(x_0, self.x, self.Q)
    # E = calculate_electric_field_cupy(x_0, x, Q)
    E = E / np.linalg.norm(E)
    x_0 = x_0 + step_size * E
    return x_0


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


def main():
    options = {
        "path_to_pqr": "../../tests/test_files/test_large.pqr",
        "center": [55.965, 46.219, 22.123],
        "x": [56.191, 48.344, 22.221],
        "y": [57.118, 46.793, 20.46],
        "n_samples": 100,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.001,
        "concur_slip": 20,
    }

    topo = Topo_calc(options)
    hist = topo.compute_topo()
    np.savetxt("hist_cpet.txt", hist)


main()
