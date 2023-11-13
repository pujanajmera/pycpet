import numpy as np
import torch
import cupy as cp

from CPET.utils.fastmath import nb_subtract, power, nb_norm, nb_cross
from CPET.utils.c_ops import Math_ops



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
    E = calculate_electric_field_dev_c_shared(x_0, x, Q)
    # E = calculate_electric_field(x_0, self.x, self.Q)
    #E = self.efield_calc(x_0, self.x, self.Q)
    # E = calculate_electric_field_cupy(x_0, x, Q)
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


def calculate_electric_field(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    # Create matrix R
    R = nb_subtract(x_0, x)
    R_sq = R**2
    r_mag_sq = np.einsum("ij->i", R_sq).reshape(-1, 1)
    r_mag_cube = np.power(r_mag_sq, 3 / 2)
    E = np.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451
    return E


def calculate_electric_field_dev_python(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    # Create matrix R
    R = nb_subtract(x_0, x)
    R_sq = R**2
    r_mag_sq = np.einsum("ij->i", R_sq).reshape(-1, 1)
    # print(R_sq.shape, r_mag_sq.shape)
    r_mag_cube = np.power(r_mag_sq, 3 / 2)
    recip_dim = 1 / r_mag_cube
    # print(R.shape, recip_dim.shape, Q.shape)
    E = np.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451
    return E

Math = Math_ops(shared_loc="../utils/math_module.so")
def calculate_electric_field_dev_c_shared(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    #if Math is None:
    #    Math = Math_ops(shared_loc="../utils/math_module.so")
    # Create matrix R
    R = nb_subtract(x_0, x)
    R_sq = R**2
    # r_mag_sq = np.einsum("ij->i", R_sq).reshape(-1, 1)
    # print(R_sq.shape)
    # print(R_sq.dtype)
    r_mag_sq = Math.einsum_ij_i(R_sq).reshape(-1, 1)
    r_mag_cube = np.power(r_mag_sq, 3 / 2)
    E = Math.einsum_operation(R, 1 / r_mag_cube, Q)
    # print(E.shape)
    # print("-")

    return E


def calculate_electric_field_gpu_torch(x_0, x, Q, device="cuda", filter=True):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (N,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    # Create matrix R
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    x_0 = torch.tensor(x_0, dtype=torch.float32, device=device)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    Q = torch.tensor(Q, dtype=torch.float32, device=device)

    R = x_0 - x
    R_sq = R**2
    r_mag_sq = torch.einsum("ij->i", R_sq).reshape(-1, 1)
    r_mag_cube = power(r_mag_sq, 3 / 2)
    E = torch.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451
    # now combine all of the above operations into one
    return E.cpu().numpy()


def calculate_electric_field_cupy(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (N,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """

    x_0 = torch.tensor(x_0)
    x = torch.tensor(x)
    Q = torch.tensor(Q)

    R = x_0 - x
    R_sq = R**2
    r_mag_sq = cp.einsum("ij->i", R_sq).reshape(-1, 1)
    r_mag_cube = power(r_mag_sq, 3 / 2)
    E = cp.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451
    # now combine all of the above operations into one
    return E.cpu().numpy()


def calculate_electric_field_cupy(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (N,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    x_0 = cp.array(x_0)
    x = cp.array(x)
    Q = cp.array(Q)
    R = x_0 - x
    R_sq = R**2
    r_mag_sq = cp.einsum("ij->i", R_sq).reshape(-1, 1)
    r_mag_cube = r_mag_sq ** (3 / 2)
    E = cp.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451
    return cp.asnumpy(E)


def compute_field_on_grid(grid_coords, x, Q):
    """
    Computes electric field at each point in a meshgrid given positions of charges.

    Takes:
        grid_coords(array): Meshgrid of points with shape (M, M, M, 3) where M is the number of points along each dimension.
        x(array): Positions of charges of shape (N, 3).
        Q(array): Magnitude and sign of charges of shape (N, 1).
    Returns:
        E(array): Electric field at each point in the meshgrid of shape (M, M, M, 3).
    """
    # Reshape meshgrid to a 2D array of shape (M*M*M, 3)
    reshaped_meshgrid = grid_coords.reshape(-1, 3)

    # Initialize an array to hold the electric field values
    E = np.zeros_like(reshaped_meshgrid, dtype=float)

    # Calculate the electric field at each point in the meshgrid
    for i, x_0 in enumerate(reshaped_meshgrid):
        # Create matrix R
        R = nb_subtract(x_0, x)
        R_sq = R**2
        r_mag_sq = np.einsum("ij->i", R_sq).reshape(-1, 1)
        r_mag_cube = np.power(r_mag_sq, 3 / 2)
        E[i] = np.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451

    # Reshape E back to the shape of the original meshgrid
    E = E.reshape(*grid_coords.shape)
    return E


def calculate_field_at_point(x, Q, x_0=np.array([0, 0, 0])):
    """
    Computes electric field at each point in a meshgrid given positions of charges.

    Takes:
        x(array): Positions of charges of shape (N, 3).
        Q(array): Magnitude and sign of charges of shape (N, 1).
        point(array): Point at which to calculate the field of shape (1, 3).
    Returns:
        E(array): Electric field at each point in the meshgrid of shape (M, M, M, 3).
    """

    # Initialize an array to hold the electric field values

    # Create matrix R
    R = nb_subtract(x_0, x)
    R_sq = R**2
    r_mag_sq = np.einsum("ij->i", R_sq).reshape(-1, 1)
    r_mag_cube = np.power(r_mag_sq, 3 / 2)
    E = np.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q) * 14.3996451
    return E


def curv(v_prime, v_prime_prime, eps=10e-6):
    """
    Computes curvature of the streamline at a given point
    Takes
        v_prime(array) - first derivative of streamline at the point of shape (1,3)
        v_prime_prime(array) - second derivative of streamline at the point of shape (1,3)
    Returns
        curvature(float) - the curvature
    """
    curvature = np.linalg.norm(np.cross(v_prime, v_prime_prime)) / (
        eps + (np.linalg.norm(v_prime) ** 3)
    )
    return curvature


def curv_dev(v_prime, v_prime_prime):
    """
    Computes curvature of the streamline at a given point
    Takes
        v_prime(array) - first derivative of streamline at the point of shape (1,3)
        v_prime_prime(array) - second derivative of streamline at the point of shape (1,3)
    Returns
        curvature(float) - the curvature of the streamline
    """

    curvature = nb_norm(nb_cross(v_prime, v_prime_prime)) / (
        10e-6 + nb_norm(v_prime) ** 3
    )

    return curvature


def compute_curv_and_dist(
    x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
):
    """
    Computes mean curvature at beginning and end of streamline and the Euclidian distance between beginning and end of streamline
    Takes
        x_init(array) - initial point of streamline of shape (1,3)
        x_init_plus(array) - initial point of streamline with one propagation of shape (1,3)
        x_init_plus_plus(array) - initial point of streamline with two propagations of shape (1,3)
        x_0(array) - final point of streamline of shape (1,3)
        x_0_plus(array) - final point of streamline with one propagation of shape (1,3)
        x_0_plus_plus(array) - final point of streamline with two propagations of shape (1,3)
    Returns
        dist(float) - Euclidian distance between beginning and end of streamline
        curv_mean(float) - mean curvature between beginning and end of streamline
    """
    curv_init = curv(x_init_plus - x_init, x_init_plus_plus - 2 * x_init_plus + x_init)
    curv_final = curv(x_0_plus - x_0, x_0_plus_plus - 2 * x_0_plus + x_0)
    curv_mean = (curv_init + curv_final) / 2
    dist = np.linalg.norm(x_init - x_0)
    return [dist, curv_mean]


def compute_curv_and_dist_dev(
    x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus
):
    """
    Computes mean curvature at beginning and end of streamline and the Euclidian distance between beginning and end of streamline
    Takes
        x_init(array) - initial point of streamline of shape (1,3)
        x_init_plus(array) - initial point of streamline with one propagation of shape (1,3)
        x_init_plus_plus(array) - initial point of streamline with two propagations of shape (1,3)
        x_0(array) - final point of streamline of shape (1,3)
        x_0_plus(array) - final point of streamline with one propagation of shape (1,3)
        x_0_plus_plus(array) - final point of streamline with two propagations of shape (1,3)
    Returns
        dist(float) - Euclidian distance between beginning and end of streamline
        curv_mean(float) - mean curvature between beginning and end of streamline
    """
    curv_init = curv_dev(
        x_init_plus - x_init, x_init_plus_plus - 2 * x_init_plus + x_init
    )
    curv_final = curv_dev(x_0_plus - x_0, x_0_plus_plus - 2 * x_0_plus + x_0)
    curv_mean = (curv_init + curv_final) / 2
    dist = nb_norm(x_init - x_0)
    return [dist, curv_mean]


def Inside_Box(local_point, dimensions):
    """
    Checks if a streamline point is inside a box
    Takes
        local_point(array) - current local point of shape (1,3)
        dimensions(array) - L, W, H of box of shape (1,3)
    Returns
        is_inside(bool) - whether the point is inside the box
    """
    # Convert lists to numpy arrays
    half_length, half_width, half_height = dimensions
    # Check if the point lies within the dimensions of the box
    is_inside = (
        -half_length <= local_point[0] <= half_length
        and -half_width <= local_point[1] <= half_width
        and -half_height <= local_point[2] <= half_height
    )
    return is_inside
