import numpy as np
import torch
import pkg_resources
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform
from CPET.utils.gpu import calculate_electric_field_torch_batch_gpu
from scipy.stats import iqr
import time
from tqdm import tqdm
import gc
import sys
import os
from kneed import KneeLocator
import tensorly as tl
from tensorly.decomposition import parafac

import importlib.util, ctypes

# Find name of c-shared library (system dependent!)
spec = importlib.util.find_spec("CPET.utils.math_module")
if spec is None or spec.origin is None:
    raise ImportError(
        "Could not find the c-shared library 'math_module'. Please ensure it is installed correctly."
    )
module_path = spec.origin

from CPET.utils.fastmath import nb_subtract, nb_norm, nb_cross
from CPET.utils.c_ops import Math_ops

Math = Math_ops(shared_loc=module_path)


def propagate_topo(x_0, x, Q, step_size, debug=False):
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
    # Compute field
    E = calculate_electric_field_base(x_0, x, Q)
    # if np.linalg.norm(E) > epsilon:
    E = E / (np.linalg.norm(E))
    x_0 = x_0 + step_size * E
    return x_0


def propagate_topo_dev(x_0, x, Q, step_size, debug=False):
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
    # Compute field
    E = calculate_electric_field_dev_c_shared(x_0, x, Q)
    # if np.linalg.norm(E) > epsilon:
    E = E / (np.linalg.norm(E))
    x_0 = x_0 + step_size * E
    return x_0


def propagate_topo_dev_batch(x_0_list, x, Q, step_size, mask_list=None):
    """
    Propagates position based on normalized electric field at a given point
    Takes
        x_0_list(array) - position to propagate based on field at that point of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
        step_size(float) - size of streamline step to take when propagating, real and positive
        mask_list(list) - list of bools to mask the points that are outside the box
    Returns
        x_0 - new position on streamline after propagation via electric field
    """

    if mask_list is None:
        E_full = calculate_electric_field_dev_c_shared_batch(x_0_list, x, Q)
        # print("passed first iter")
    else:
        # if theres a mask, only calculate the field for the points that are inside the box
        # filter x_0_list on the mask

        x_0_list_filtered = [x_0 for x_0, mask in zip(x_0_list, mask_list) if not mask]
        E = calculate_electric_field_dev_c_shared_batch(x_0_list_filtered, x, Q)
        # expand E to the original size E is 0 for masked points
        # print("mask list: {}".format(mask_list))
        # print("E: {}".format(E))
        E_full = []
        # print("E: {}".format(E))
        # print("type {}".format(type(E[0])))
        ind_filtered = 0
        for ind, x in enumerate(x_0_list):
            if mask_list[ind] == False:
                E_full.append(E[ind_filtered])
                ind_filtered += 1
            else:
                E_full.append([0.0, 0.0, 0.0])

    assert len(E_full) == len(x_0_list), "efull len is funky"

    # print("efull: {}".format(E_full))
    E_list = [e / (np.linalg.norm(e)) for e in E_full]
    x_0_list = [x_0 + step_size * e for x_0, e in zip(x_0_list, E_list)]
    # print(x_0_list)
    return x_0_list


def initialize_box_points_random(
    center, x, y, dimensions, n_samples, max_steps=10, dtype="float32"
):
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
    print("... initializing box points randomly")
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
    if dtype == "float32":
        random_points_local = random_points_local.astype(np.float32)
    # Convert these points back to the global coordinate system
    transformation_matrix = np.column_stack(
        [x_unit, y_unit, z_unit]
    ).T  # Each column is a unit vector

    random_max_samples = np.random.randint(1, max_steps, n_samples)
    return random_points_local, random_max_samples, transformation_matrix, max_steps


def initialize_box_points_uniform(
    center,
    x,
    y,
    N_cr,
    dimensions,
    dtype="float32",
    max_steps=10,
    ret_rand_max=False,
    inclusive=True,
    seed=None,
):
    """
    Initializes random points in box centered at the origin
    Takes
        center(array) - center of box of shape (1,3)
        x(array) - point to create x axis from center of shape (1,3)
        y(array) - point to create x axis from center of shape (1,3)
        N_cr
        dimensions(array) - L, W, H of box of shape (1,3)
        n_samples(int) - number of samples to compute
        step_size(float) - step_size of box
    Returns
        random_points_local(array) - array of random starting points in the box of shape (n_samples,3)
        transformation_matrix(array) - matrix that contains the basis vectors for the box of shape (3,3)
    """
    # Convert lists to numpy arrays
    print("... initializing box points uniformly")

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

    transformation_matrix = np.column_stack(
        [x_unit, y_unit, z_unit]
    ).T  # Each column is a unit vector
    print(N_cr)
    # construct a grid of points in the box - lengths are floats
    if inclusive:
        x_coords = np.linspace(-half_length, half_length, N_cr[0] + 1)
        y_coords = np.linspace(-half_width, half_width, N_cr[1] + 1)
        z_coords = np.linspace(-half_height, half_height, N_cr[2] + 1)
    else:
        x_coords = np.linspace(-half_length, half_length, N_cr[0] + 1, endpoint=False)[
            1:
        ]
        y_coords = np.linspace(-half_width, half_width, N_cr[1] + 1, endpoint=False)[1:]
        z_coords = np.linspace(-half_height, half_height, N_cr[2] + 1, endpoint=False)[
            1:
        ]

    x_mesh, y_mesh, z_mesh = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    local_coords = np.stack([x_mesh, y_mesh, z_mesh], axis=-1, dtype=dtype)
    print(local_coords.shape)
    if not seed == None:
        np.random.seed(seed)

    if ret_rand_max:
        n_samples = (
            local_coords.shape[0] * local_coords.shape[1] * local_coords.shape[2]
        )
        # print(f"max distance: {max_distance}")
        # print(f"step size: {step_size}")
        # print(f"max steps: {max_steps}")
        random_max_samples = np.random.randint(1, max_steps, n_samples)
        return local_coords, random_max_samples, transformation_matrix

    return local_coords, transformation_matrix


def calculate_electric_field_base(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges, basic version
    Takes
        x_0(array) - position to compute field at of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    # Create matrix R
    R = np.subtract(x_0, x)  # Shape (1, N, 3)
    denom = np.linalg.norm(R) ** 3  # Shape (1, N)
    E_vec = R * (1 / denom).reshape(-1, 1) * Q * 14.3996451  # Shape (1, N, 3)
    return E_vec


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
    # if Math is None:
    #    Math = Math_ops(shared_loc="../utils/math_module.so")
    # Create matrix R
    # print("subtract")

    R = nb_subtract(x_0, x)  # right

    R_sq = R**2  # right
    R_sq = R_sq.astype(np.float32)
    r_mag_sq = Math.einsum_ij_i(R_sq).reshape(-1, 1)
    r_mag_cube = np.power(r_mag_sq, -3 / 2)
    E = Math.einsum_operation(R, r_mag_cube, Q)
    # print(E.shape)
    # print("-")
    # print("end efield calc")
    return E


def calculate_electric_field_c_shared_full_alt(x_0, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    x_0 = x_0.astype(np.float32)
    x = x.astype(np.float32)
    Q = Q.astype(np.float32)
    E = Math.calc_field(x_0=x_0, x=x, Q=Q)
    return E


def calculate_esp_c_shared_full(x_0, x, Q):
    """
    Computes electrostatic potential at a point given positions of charges
    Takes
        x_0(array) - position to compute field at of shape (1,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        ESP(float) - electrostatic potential at the point
    """
    ESP = Math.calc_esp_base(x_0=x_0, x=x, Q=Q)
    return ESP


def calculate_thread_c_shared(x_0, n_iter, x, Q, step_size, dimensions):
    result = Math.thread_operation(
        x_0=x_0, n_iter=n_iter, x=x, Q=Q, step_size=step_size, dimensions=dimensions
    )
    return result


def calculate_electric_field_dev_c_shared_batch(x_0_list, x, Q):
    """
    Computes electric field at a point given positions of charges
    Takes
        x_0(list of arrays) - position to compute field at of shape [n by (1,3)]
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E(array) - electric field at the point of shape (1,3)
    """
    R_list = [nb_subtract(x_0, x) for x_0 in x_0_list]
    batch_size = len(R_list)
    R_list = np.array(R_list, dtype="float32")  # Shape (n, N, 3)
    R_sq = np.array([R**2 for R in R_list], dtype="float32")  # Shape (n, N, 3)
    r_mag_sq = Math.einsum_ij_i_batch(R_sq)  # .reshape(-1, 1)
    r_mag_cube = np.power(r_mag_sq, 3 / 2)
    recip_r_mag = 1 / r_mag_cube  # Shape (n, N)
    # print("recip r {}".format(recip_r_mag_list))
    E_list = (
        Math.einsum_operation_batch(R_list, recip_r_mag, Q, batch_size) * 14.3996451
    )
    # print("passed einsum op")
    # print(E.shape)
    # print("-")
    # print("Elist: {}".format(E_list))
    return E_list


def calculate_electric_field_gpu_for_test(x_0, x, Q, device="cuda"):
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

    if x_0.dim() == 1:
        x_0 = x_0.unsqueeze(0)

    E = calculate_electric_field_torch_batch_gpu(x_0, x, Q)

    if E.dim() == 2 and E.size(0) == 1:
        E = E.squeeze(0)

    return E.cpu().numpy()


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
    x_0 = grid_coords.reshape(-1, 3)

    # Compute the field using loop c-shared library
    E = Math.compute_looped_field(x_0, x, Q)

    return E


def compute_ESP_on_grid(grid_coords, x, Q):
    """
    Computes electrostatic potential at each point in a meshgrid given positions of charges.

    Takes:
        grid_coords(array): Meshgrid of points with shape (M, M, M, 3) where M is the number of points along each dimension.
        x(array): Positions of charges of shape (N, 3).
        Q(array): Magnitude and sign of charges of shape (N, 1).
    Returns:
        point_ESP_concat(array): Position and electrostatic potential at each point in the meshgrid of shape (M**3, 4).
    """
    # Reshape meshgrid to a 2D array of shape (M*M*M, 3)
    reshaped_meshgrid = grid_coords.reshape(-1, 3)
    # Initialize an array to hold the electric field values
    ESP = np.zeros((reshaped_meshgrid.shape[0], 1), dtype=float)
    print(ESP.shape)

    # Calculate the electric field at each point in the meshgrid
    for i, x_0 in enumerate(reshaped_meshgrid):
        ESP[i] = calculate_esp_c_shared_full(x_0, x, Q)
        if x_0[0] == 0 and x_0[1] == 0 and x_0[2] == 0:
            print(f"Center esp: {ESP[i]}")

    point_ESP_concat = np.concatenate((reshaped_meshgrid, ESP), axis=1)

    return point_ESP_concat.astype(np.half)


def calculate_field_at_point(x, Q, x_0=np.array([0, 0, 0])):
    """
    Computes electric field at each point in a meshgrid given positions of charges.

    Takes:
        x(array): Positions of charges of shape (N, 3).
        Q(array): Magnitude and sign of charges of shape (N, 1).
        point(array): Point at which to calculate the field of shape (1, 3).
    Returns:
        E(array): Electric field at each point in the meshgrid of shape (1, 3).
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
    # if np.linalg.norm(v_prime) ** 3 < eps:
    #    denominator = eps
    # else:
    #    denominator = np.linalg.norm(v_prime) ** 3

    denominator = np.linalg.norm(v_prime, axis=0) ** 3
    curvature = np.linalg.norm(np.cross(v_prime, v_prime_prime), axis=0)

    if denominator == 0:
        return curvature / eps
    else:
        return curvature / denominator


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
    dist = np.linalg.norm(x_init - x_0, axis=-1)
    return [dist, curv_mean]


def inside_box_mask(points, dimensions):
    """
    Masked version of Inside_Box for more than one point
    """
    half_length, half_width, half_height = dimensions
    cond_x = (points[:, 0] >= -half_length) & (points[:, 0] <= half_length)
    cond_y = (points[:, 1] >= -half_width) & (points[:, 1] <= half_width)
    cond_z = (points[:, 2] >= -half_height) & (points[:, 2] <= half_height)
    return cond_x & cond_y & cond_z


def Inside_Box(local_point, dimensions):
    """
    Checks if a streamline point is inside a box
    Takes
        local_point(array) - current local point of shape (1,3)
        dimensions(array) - L, W, H of box of shape (1,3)
    Returns
        is_inside(bool) - whether the point is inside the box
    """
    # print("inside box")
    # Convert lists to numpy arrays
    half_length, half_width, half_height = dimensions
    # Check if the point lies within the dimensions of the box
    is_inside = (
        -half_length <= local_point[0] <= half_length
        and -half_width <= local_point[1] <= half_width
        and -half_height <= local_point[2] <= half_height
    )
    return is_inside


def make_histograms(topo_files, plot=False):
    histograms = []

    # First pass: Calculate total number of data points
    len_list = np.zeros(len(topo_files), dtype=int)
    start_time = time.time()
    # Use tqdm instead of original for loop
    for idx, topo_file in tqdm(enumerate(topo_files), total=len(topo_files)):
        with open(topo_file) as topology_data:
            line_count = sum(1 for line in topology_data if not line.startswith("#"))
            len_list[idx] = line_count
    total_points = np.sum(len_list)
    end_time = time.time()
    print(f"Time taken to count data points: {end_time - start_time:.2f} seconds")

    # Initialize NumPy arrays with the total size
    dist_list = np.zeros(total_points)
    curv_list = np.zeros(total_points)
    topo_data_indices = np.zeros(
        len(topo_files) + 1, dtype=int
    )  # To store start indices

    # Second pass: Read data into NumPy arrays
    start_time = time.time()
    current_index = 0
    for idx, topo_file in tqdm(enumerate(topo_files), total=len(topo_files)):
        num_points = len_list[idx]
        distances = np.zeros(num_points)
        curvatures = np.zeros(num_points)
        with open(topo_file) as topology_data:
            point_idx = 0
            for line in topology_data:
                if line.startswith("#"):
                    continue
                line = line.strip().split()
                distances[point_idx] = float(line[0])
                curvatures[point_idx] = float(line[1])
                point_idx += 1

        if distances.size != curvatures.size:
            raise ValueError(
                f"Length of distances and curvatures do not match for {topo_file}"
            )

        # Store data in the main arrays
        dist_list[current_index : current_index + num_points] = distances
        curv_list[current_index : current_index + num_points] = curvatures

        # Store the start index for this file's data
        topo_data_indices[idx] = current_index
        current_index += num_points

    # Append the final index
    topo_data_indices[-1] = current_index
    end_time = time.time()
    print(f"Time taken to read data into arrays: {end_time - start_time:.2f} seconds")

    # Check if all files have the same length
    len_list_equality = np.all(len_list == len_list[0]) if len_list.size > 0 else True
    if not len_list_equality:
        warnings.warn(
            f"Topologies provided are of different sizes, using the mean value of {np.mean(len_list)} "
            f"to represent binning for {len_list}"
        )
        len_dist_curv = np.mean(len_list)
    else:
        len_dist_curv = len_list[0]

    max_distance = np.max(dist_list)
    max_curvature = np.max(curv_list)
    min_distance = np.min(dist_list)
    min_curvature = np.min(curv_list)

    distance_binres = 2 * iqr(dist_list) / (len_dist_curv ** (1 / 3))
    curv_binres = 2 * iqr(curv_list) / (len_dist_curv ** (1 / 3))

    # distance_binres = 0.02
    # curv_binres = 0.02

    print(f"Distance bin resolution: {distance_binres}")
    print(f"Curvature bin resolution: {curv_binres}")

    print(f"Max distance: {max_distance}")
    print(f"Min distance: {min_distance}")
    print(f"Max curvature: {max_curvature}")
    print(f"Min curvature: {min_curvature}")

    # Calculate number of bins
    distance_nbins = int((max_distance - min_distance) / distance_binres)
    curvature_nbins = int((max_curvature - min_curvature) / curv_binres)

    start_time = time.time()
    # Make histograms
    for idx in tqdm(range(len(topo_files))):
        with open(topo_files[idx]) as topology_data:
            distances = []
            curvatures = []
            for line in topology_data:
                if line.startswith("#"):
                    continue

                line = line.split()
                distances.append(float(line[0]))
                curvatures.append(float(line[1]))

        # Compute the 2D histogram
        a, b, c = np.histogram2d(
            distances,
            curvatures,
            bins=[distance_nbins, curvature_nbins],
            range=[[min_distance, max_distance], [min_curvature, max_curvature]],
        )
        del distances
        del curvatures
        NormConstant = np.sum(a)
        actual = a / NormConstant

        histograms.append(actual.flatten())
    end_time = time.time()
    print(
        f"Time taken to parse topology files into histograms: {end_time - start_time:.2f} seconds"
    )
    return np.array(histograms)


def make_histograms_mem(topo_files, output_dir, plot=False):
    histograms = []

    # Calculate reasonable maximum distances and curvatures
    dist_list = []
    curv_list = []
    len_list = []
    topo_list = []
    start_time = time.time()
    for topo_file in topo_files:
        curvatures, distances = [], []
        print(topo_file)
        sys.stdout.flush()
        time.sleep(1.0)
        with open(topo_file) as topology_data:
            for line in topology_data:
                if line.startswith("#"):
                    continue
                # strip the line of any leading or trailing whitespace
                line = line.strip()
                # if the line has a comma split by comma
                if "," in line:
                    line = line.split(",")
                else:
                    line = line.split()
                # print(line)
                distances.append(float(line[0]))
                curvatures.append(float(line[1]))
        # print(max(distances),max(curvatures))
        if len(distances) != len(curvatures):
            ValueError(
                f"Length of distances and curvatures do not match for {topo_file}"
            )
        else:
            len_list.append(len(distances))
        topo_list.append((distances, curvatures))
        dist_list.extend(distances)
        curv_list.extend(curvatures)
        del distances
        del curvatures
    gc.collect()
    end_time = time.time()
    print(f"Time taken to parse topology files: {end_time - start_time:.2f} seconds")
    sys.stdout.flush()
    time.sleep(1.0)
    len_list_equality = all(x == len_list[0] for x in len_list) if len_list else True
    if not len_list_equality:
        warnings.warn(
            f"Topologies provided are of different sizes, using the mean value of {np.mean(len_list)} to represent binning for {len_list}"
        )
        len_dist_curv = np.mean(len_list)
    else:
        len_dist_curv = len_list[0]

    max_distance = max(dist_list)
    max_curvature = max(curv_list)
    min_distance = min(dist_list)
    min_curvature = min(curv_list)

    distance_binres = 2 * iqr(dist_list) / (len_dist_curv ** (1 / 3))
    curv_binres = 2 * iqr(curv_list) / (len_dist_curv ** (1 / 3))

    del dist_list
    del curv_list

    # distance_binres = 0.02
    # curv_binres = 0.02

    print(f"Distance bin resolution: {distance_binres}")
    print(f"Curvature bin resolution: {curv_binres}")

    print(f"Max distance: {max_distance}")
    print(f"Min distance: {min_distance}")
    print(f"Max curvature: {max_curvature}")
    print(f"Min curvature: {min_curvature}")
    sys.stdout.flush()
    time.sleep(1.0)
    # Need 0.02A resolution for max_distance
    distance_nbins = int((max_distance - min_distance) / distance_binres)
    # distance_nbins = 200
    # Need 0.02A resolution for max_curvature
    curvature_nbins = int((max_curvature - min_curvature) / curv_binres)
    # curvature_nbins = 200

    start_time = time.time()
    # Make histograms
    hist_list = []
    for i, topo_file in enumerate(topo_files):
        distances, curvatures = topo_list[i]
        # print(f"Plotting histo for {topo_file}")
        # bins is number of histograms bins in x and y direction
        # range gives xrange, yrange for the histogram
        a, b, c, q = plt.hist2d(
            distances,
            curvatures,
            bins=(distance_nbins, curvature_nbins),
            range=[[min_distance, max_distance], [min_curvature, max_curvature]],
            density=True,
            cmap="jet",
        )
        NormConstant = 0
        for j in a:
            for m in j:
                NormConstant += m
        # print(NormConstant)
        actual = []
        for j in a:
            actual.append([m / NormConstant for m in j])

        actual = np.array(actual)
        outfile = os.path.join(
            output_dir, f"{topo_files[i].split('/')[-1][:-4]}_hist.npy"
        )
        np.save(outfile, actual.flatten())
        hist_list.append(outfile)
        del actual
        if plot:
            plt.show()
    end_time = time.time()
    print(
        f"Time taken to parse topology files into histograms: {end_time - start_time:.2f} seconds"
    )
    return hist_list


def make_fields(field_files):
    fields = []
    for field_file in field_files:
        field = []
        with open(field_file) as field_data:
            for line in field_data:
                if line.startswith("#"):
                    continue
                line = line.split()
                field.append([float(line[3]), float(line[4]), float(line[5])])
        fields.append(np.array(field))
    return fields


def make_single_4d_tensor(
    field_file, uniques=None, N=[0, 0, 0], type="field", init=False
):
    with open(field_file, "r") as file:
        lines = file.readlines()

    # Skip the first 7 lines (header)
    if type == "field":
        data_lines = lines[7:]
    elif type == "esp":
        data_lines = lines[:]

    # Initialize lists to store the parsed data
    x_coords, y_coords, z_coords = [], [], []
    if type == "field":
        ex, ey, ez = [], [], []
    elif type == "esp":
        e = []

    # Loop through the remaining lines and extract the data
    for line in data_lines:
        cols = line.split()
        # print(cols)
        x_coords.append(float(cols[0]))
        y_coords.append(float(cols[1]))
        z_coords.append(float(cols[2]))
        if type == "field":
            ex.append(float(cols[3]))
            ey.append(float(cols[4]))
            ez.append(float(cols[5]))
        elif type == "esp":
            e.append(float(cols[3]))

    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)

    if type == "field":
        ex = np.array(ex)
        ey = np.array(ey)
        ez = np.array(ez)
    elif type == "esp":
        e = np.array(e)

    if init == False:
        if N == [0, 0, 0]:
            ValueError("Please provide the dimensions of the grid if init is false")
        if type == "field":
            tensor_4d = np.zeros((N[0], N[1], N[2], 3))
        elif type == "esp":
            tensor_4d = np.zeros((N[0], N[1], N[2], 1))
        unique_x, unique_y, unique_z = uniques
    else:
        # Find unique values of X, Y, Z to determine grid dimensions
        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)
        unique_z = np.unique(z_coords)
        print(unique_x, unique_y, unique_z)

        if N == [0, 0, 0]:
            print(
                "Init is true, ignoring N=[0,0,0] and writing from input field/esp file"
            )
        # Determine grid dimensions
        Nx, Ny, Nz = len(unique_x), len(unique_y), len(unique_z)

        # Initialize a 4D tensor of shape [Nx, Ny, Nz, 3] for (Ex, Ey, Ez)
        if type == "field":
            tensor_4d = np.zeros((Nx, Ny, Nz, 3))
        elif type == "esp":
            tensor_4d = np.zeros((Nx, Ny, Nz, 1))
            print(tensor_4d.shape)

    # Populate the tensor with the field data
    for i in range(len(x_coords)):
        ix = np.where(unique_x == x_coords[i])[0][0]
        iy = np.where(unique_y == y_coords[i])[0][0]
        iz = np.where(unique_z == z_coords[i])[0][0]

        if type == "field":
            tensor_4d[ix, iy, iz, 0] = ex[i]  # Ex
            tensor_4d[ix, iy, iz, 1] = ey[i]  # Ey
            tensor_4d[ix, iy, iz, 2] = ez[i]  # Ez
        elif type == "esp":
            tensor_4d[ix, iy, iz, 0] = e[i]  # ESP

    if init == True:
        return tensor_4d, [Nx, Ny, Nz], (unique_x, unique_y, unique_z)

    return tensor_4d


def make_5d_tensor(field_files, type="field"):
    first_field = field_files[0]
    if type == "field":
        _, N, uniques = make_single_4d_tensor(first_field, init=True)
        tensor_5d = np.zeros((len(field_files), N[0], N[1], N[2], 3))
        for i, field_file in enumerate(field_files):
            tensor_5d[i] = make_single_4d_tensor(
                field_file, uniques=uniques, N=N, type="field"
            )
    elif type == "esp":
        _, N, uniques = make_single_4d_tensor(first_field, type="esp", init=True)
        tensor_5d = np.zeros((len(field_files), N[0], N[1], N[2], 1))
        for i, field_file in enumerate(field_files):
            print(i)
            tensor_5d[i] = make_single_4d_tensor(
                field_file, uniques=uniques, N=N, type="esp"
            )
    else:
        ValueError(
            "Please provide the correct type of tensor clustering method, either 'field' or 'esp'"
        )

    return tensor_5d


def distance_numpy(hist1, hist2):
    a = (hist1 - hist2) ** 2
    b = hist1 + hist2
    return np.sum(np.divide(a, b, out=np.zeros_like(a), where=b != 0)) / 2.0


def construct_distance_matrix_mem(hist_file_list):
    """
    Memory-efficient implementation
    """
    start_time = time.time()
    n = len(hist_file_list)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            hist1 = np.load(hist_file_list[i])
            hist2 = np.load(hist_file_list[j])
            matrix[i][j] = distance_numpy(hist1, hist2)
            matrix[j][i] = matrix[i][j]
            del hist1
            del hist2
    end_time = time.time()
    print(
        f"Time taken to generate pairwise distance matrix: {end_time - start_time:.2f} seconds"
    )
    return matrix


def construct_distance_matrix(histograms):
    from sklearn.metrics.pairwise import pairwise_distances

    start_time = time.time()
    n = len(histograms)
    matrix = pairwise_distances(histograms, metric=distance_numpy, n_jobs=-1)
    for i in range(n):
        matrix[i][i] = 0.0  # Make sure the diagonal is zero
    end_time = time.time()
    print(
        f"Time taken to generate pairwise distance matrix: {end_time - start_time:.2f} seconds"
    )
    return matrix


"""
def construct_distance_matrix_alt2(histograms):
    new_histograms = []
    start_time = time.time()
    for h in histograms:
        h_copy = np.copy(h)  # Make a copy of the histogram
        h_copy[h_copy < 0.001] = 0  # Set values less than 0.001 to zero in the
        h_copy = h_copy / np.sum(h_copy)  # Normalize the histogram
        new_histograms.append(h_copy)  # Append the modified copy to the new list
    end_time = time.time()
    print(f"Time taken to filter and renormalize histograms: {end_time - start_time:.2f} seconds")
    matrix = np.diag(np.zeros(len(new_histograms)))
    start_time = time.time()
    for i, hist1 in enumerate(new_histograms):
        for j, hist2 in enumerate(new_histograms[i + 1 :]):
            j += i + 1
            matrix[i][j] = distance_numpy(hist1, hist2)
            matrix[j][i] = matrix[i][j]
    end_time = time.time()
    print(f"Time taken to generate pairwise distance matrix: {end_time - start_time:.2f} seconds")
    return matrix
"""


def construct_distance_matrix_volume(fields):
    """
    Computes the distance matrix between vector fields using the cosine similarity
    Takes
        fields(list of arrays) - vector fields of shape (N,3)
    Returns
        matrix(array) - distance matrix between vector fields
    """
    matrix = np.diag(np.zeros(len(fields)))
    for i, field1 in enumerate(fields):
        for j, field2 in enumerate(fields[i + 1 :]):
            j += i + 1
            # Euclidean distance for vector fields
            dists = np.linalg.norm(field1 - field2, axis=-1)
            matrix[i][j] = np.sum(dists)
            matrix[j][i] = matrix[i][j]
    return matrix


def construct_distance_matrix_tensor(cp_tensor):
    factors = cp_tensor.factors
    factors_dict = {f"factor_{idx}": factor for idx, factor in enumerate(factors)}
    factor_matrix = factors[0]

    # Plotting
    # plot_rank_variation(factor_matrix)

    dist_mat = squareform(pdist(factor_matrix, metric="euclidean"))
    return dist_mat


def reduce_tensor(tensor, rank):
    # Perform CP decomposition
    cp_tensor = parafac(tensor, rank=rank, init="random", tol=1e-6, random_state=42)

    # Reconstruct the tensor from the CP decomposition
    reconstructed_tensor = tl.cp_to_tensor(cp_tensor)

    # Calculate the reconstruction error (mean squared error)
    absolute_error = mean_squared_error(
        tensor.flatten(), reconstructed_tensor.flatten()
    )

    # Calculate relative error (RMSE or another metric)
    relative_error = absolute_error / np.mean(np.abs(tensor.flatten()))

    return cp_tensor, relative_error


def determine_rank(tensor_5d, threshold, max_rank, min_rank):
    errors = []
    ranks = list(range(min_rank, max_rank + 1))

    # Loop over possible ranks and compute errors
    for rank in ranks:
        start_time = time.time()

        _, error = reduce_tensor(tensor_5d, rank)
        errors.append(error)
        print(f"Rank {rank}, Reconstruction Error: {error}")

        end_time = time.time()
        print(
            f"Time taken to do CP decomposition for rank {rank}: {end_time - start_time:.4f} seconds"
        )

    # Using elbow method
    # Use kneed to find the elbow point
    kneedle = KneeLocator(ranks, errors, curve="convex", direction="decreasing")
    elbow_rank = kneedle.elbow
    optimal_rank = 50  # Just some default value

    # Check if the reconstruction error at the elbow rank is less than tensor_threshold
    if errors[elbow_rank - 1] > threshold:
        print(
            f"Reconstruction error at elbow rank {elbow_rank} is {errors[elbow_rank - 1]}, which is > {threshold}"
        )
        print(f"Searching for the lowest rank with error < {threshold}...")

        # Find the lowest rank where the reconstruction error is < tensor_threshold
        for i, error in enumerate(errors):
            if error < threshold:
                optimal_rank = ranks[i]
                print(
                    f"Found optimal rank {optimal_rank} with reconstruction error {error}"
                )
                break
    else:
        optimal_rank = elbow_rank
        print(f"Optimal Rank based on the elbow method (kneedle): {optimal_rank}")

    return optimal_rank


def report_inside_box(calculator_object):
    """
    Reports atoms that are inside the box, not including anything
    that has been filtered (vectorized). Trick to ignore atoms more than 1A outside of box circumsphere
    (along largest side)
    """
    calc_obj_copy = (
        calculator_object  # Copy the object to avoid modifying the original object
    )
    max_len_box = calc_obj_copy.dimensions.max()
    outside_sphere = np.linalg.norm(calc_obj_copy.x - calc_obj_copy.center, axis=1) > (
        max_len_box + 1
    )  # 1A wiggle room

    for i in np.where(outside_sphere)[0]:
        # Check to see if atom is in box
        if Inside_Box(calc_obj_copy.x[i], calc_obj_copy.dimensions):
            print(
                "Atom record {}_{}_{}_{} found inside box".format(
                    calculator_object.atom_number[i],
                    calculator_object.atom_type[i],
                    calculator_object.resids[i],
                    calculator_object.residue_number[i],
                )
            )
            print("Corresponding protein: {}".format(calculator_object.path_to_pdb))
