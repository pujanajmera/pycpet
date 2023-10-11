import numpy as np
import time
from CPET.utils.calculator import (
    calculate_electric_field,
    curv,
    compute_curv_and_dist,
    Inside_Box,
)
from CPET.utils.parser import parse_pqr


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
    E = calculate_electric_field(x_0, x, Q)  # Compute field
    E = E / np.linalg.norm(E)
    x_0 = x_0 + step_size * E
    return x_0


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


def main():
    options = {
        "path_to_pqr": "../../tests/test_files/test.pqr",
        "center": [55.965, 46.219, 22.123],
        "x": [56.191, 48.344, 22.221],
        "y": [57.118, 46.793, 20.46],
        "n_samples": 100,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.001,
    }
    x, Q = parse_pqr(options["path_to_pqr"])
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
