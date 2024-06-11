import numpy as np
import time
import torch
import torch.jit as jit
from typing import Tuple
from CPET.utils.parser import parse_pqr
from torch.profiler import profile, record_function, ProfilerActivity


def check_tensor(x, name="Tensor"):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"{name} contains NaN or Inf")
        return True
    return False


'''
#@profile
@torch.jit.script
def calculate_electric_field_torch_batch_gpu(x_0: torch.Tensor, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    N = x_0.size(0)
    E = torch.zeros(N, 3, device=x_0.device)

    for start in range(0, N, 1000):
        end = min(start + 1000, N)
        #x_0_batch = x_0[start:end]
        #R = x_0_batch.unsqueeze(1) - x.unsqueeze(0)
        R = x_0[start:end].unsqueeze(1) - x.unsqueeze(0)
        r_mag_cube = torch.norm(R, dim=-1, keepdim=True).pow(3)
        E[start:end] = torch.einsum("ijk,ijk,ijk->ik", R, 1/r_mag_cube, Q) * 14.3996451

    return E

@torch.jit.script
def propagate_topo_matrix_gpu(path_matrix: torch.Tensor,i: torch.Tensor, x: torch.Tensor, Q: torch.Tensor, step_size: torch.Tensor) -> torch.Tensor:
#def propagate_topo_matrix_gpu(path_matrix, i, x, Q, step_size):
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
    path_matrix_prior = path_matrix[int(i)]
    E = calculate_electric_field_torch_batch_gpu(path_matrix_prior, x, Q)  # Compute field
    path_matrix[i+1] = path_matrix_prior + step_size* E / torch.norm(E, dim=-1, keepdim=True)
    return path_matrix
'''


@torch.jit.script
def propagate_topo_matrix_gpu(
    path_matrix: torch.Tensor,
    i: torch.Tensor,
    x: torch.Tensor,
    Q: torch.Tensor,
    step_size: torch.Tensor,
) -> torch.Tensor:
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
    # torch.autograd.set_detect_anomaly(True)

    path_matrix_prior = path_matrix[int(i)]

    # check_tensor(path_matrix_prior, "Path Matrix Prior")
    N = path_matrix_prior.size(0)
    # E = torch.zeros(N, 3, device=path_matrix_prior.device, dtype=torch.float16)
    E = torch.zeros(N, 3, device=path_matrix_prior.device, dtype=torch.float32)

    for start in range(0, N, 100):
        end = min(start + 100, N)
        # x_0_batch = x_0[start:end]
        # R = x_0_batch.unsqueeze(1) - x.unsqueeze(0)
        R = path_matrix_prior[start:end].unsqueeze(1) - x.unsqueeze(0)
        # R = R.to(torch.float16)
        # print(R.dtype)
        # check_tensor(R, "R")
        r_mag_cube = torch.norm(R, dim=-1, keepdim=True).pow(-3)
        # check_tensor(r_mag_cube, "R Mag Cube")
        # E[start:end] = torch.einsum("ijk,ijk,ijk->ik", R, 1/r_mag_cube, Q) * 14.3996451
        E[start:end] = (R * r_mag_cube * Q).sum(dim=1) * 14.3996451
        # check_tensor(E, "Electric Field")

    path_matrix[i + 1] = path_matrix_prior + step_size * E / torch.norm(
        E, dim=-1, keepdim=True
    )

    return path_matrix


@torch.jit.script
def curv_mat_gpu(v_prime: torch.Tensor, v_prime_prime: torch.Tensor) -> torch.Tensor:
    """
    Computes curvature of the streamline at a given point
    Takes
        v_prime(array) - first derivative of streamline at the point of shape (N,3)
        v_prime_prime(array) - second derivative of streamline at the point of shape (N,3)
    Returns
        curvature(float) - the curvature
    """
    curvature = (
        torch.norm(torch.cross(v_prime, v_prime_prime), dim=-1)
        / torch.norm(v_prime, dim=-1) ** 3
    )
    return curvature


@torch.jit.script
def compute_curv_and_dist_mat_gpu(
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
    curv_init = curv_mat_gpu(
        x_init_plus - x_init, x_init_plus_plus - 2 * x_init_plus + x_init
    )
    curv_final = curv_mat_gpu(x_0_plus - x_0, x_0_plus_plus - 2 * x_0_plus + x_0)
    curv_mean = (curv_init + curv_final) / 2
    # print(x_init)
    # print(x_0)
    dist = torch.norm(x_init - x_0, dim=-1)
    return dist, curv_mean


@torch.jit.script
def Inside_Box_gpu(local_points, dimensions):
    """
    Checks if a streamline point is inside a box
    Takes
        local_point(array) - current local points of shape (M,N,3)
        dimensions(array) - L, W, H of box of shape (1,3)
    Returns
        is_inside(bool) - whether the point is inside the box
    """
    # Convert lists to numpy arrays
    half_length, half_width, half_height = dimensions[0], dimensions[1], dimensions[2]
    # Check if the point lies within the dimensions of the box
    is_inside = (
        (local_points[..., 0] >= -half_length)
        & (local_points[..., 0] <= half_length)
        & (local_points[..., 1] >= -half_width)
        & (local_points[..., 1] <= half_width)
        & (local_points[..., 2] >= -half_height)
        & (local_points[..., 2] <= half_height)
    )
    return is_inside


def initialize_streamline_grid_gpu(
    center, x, y, dimensions, num_per_dim, step_size, dtype_str="float64"
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
        dtype_str(str) - data type of the points
    Returns
        random_points_local(array) - array of random starting points in the box of shape (n_samples,3)
        random_max_samples(array) - array of maximum sample number for each streamline of shape (n_samples, 1)
        transformation_matrix(array) - matrix that contains the basis vectors for the box of shape (3,3)
    """

    N_cr = num_per_dim
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
    """
    # Generate random samples in the local coordinate system of the box
    random_x = np.random.uniform(-half_length, half_length, N)
    random_y = np.random.uniform(-half_width, half_width, N)
    random_z = np.random.uniform(-half_height, half_height, N)
    # Each row in random_points_local corresponds to x, y, and z coordinates of a point in the box's coordinate system
    random_points_local = np.column_stack([random_x, random_y, random_z])
    """
    # Calculate the number of points along each dimension

    x_coords = np.linspace(-half_length, half_length, N_cr + 1, endpoint=False)[1:]
    y_coords = np.linspace(-half_width, half_width, N_cr + 1, endpoint=False)[1:]
    z_coords = np.linspace(-half_height, half_height, N_cr + 1, endpoint=False)[
        1:
    ]  # Use meshgrid to create coordinates
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    points = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=-1)

    # Convert these points back to the global coordinate system
    transformation_matrix = np.column_stack(
        [x_unit, y_unit, z_unit]
    ).T  # Each column is a unit vector
    max_distance = 2 * np.linalg.norm(
        np.array(dimensions)
    )  # Define maximum sample limit as 2 times the diagonal
    M = round(max_distance / step_size)
    np.random.seed(42)
    random_max_samples = torch.tensor(np.random.randint(1, M, N_cr**3)).cuda()
    print(random_max_samples)
    np.savetxt("points.txt", points)
    path_matrix = np.zeros((M + 2, N_cr**3, 3))
    path_matrix[0] = torch.tensor(points)
    path_filter = generate_path_filter_gpu(
        random_max_samples, torch.tensor([M + 2], dtype=torch.int64).cuda()
    )
    print(M, N_cr**3)
    return path_matrix, transformation_matrix, M, path_filter, random_max_samples


# @torch.jit.script
def generate_path_filter_gpu(arr, M):
    # Initialize the matrix with zeros

    mat = torch.zeros((len(arr), int(M)), dtype=torch.int64, device="cuda")

    # Iterate over the array
    for i, value in enumerate(arr):
        # Set the values to 1 up to and including 2 after the entry value
        if value != -1:
            mat[i, : value + 2] = 1
        else:
            mat[i] = 1
    # return np.expand_dims(mat.T,axis=2)
    return torch.unsqueeze(mat.permute(1, 0), dim=2)


"""
@torch.jit.script
def generate_path_filter_gpu(arr: torch.Tensor, M: int) -> torch.Tensor:
    # Initialize the matrix with zeros
    mat = torch.zeros((len(arr), M), dtype=torch.int64, device='cuda')

    # Compute indices where value is not -1
    valid_indices = arr != -1

    # Compute the range limits for each valid index
    limits = arr[valid_indices] + 2  # +2 to include two more elements

    # Use broadcasting to create the filter mask
    # Create a row vector from 0 to M-1 and compare it against column vector 'limits'
    index_matrix = torch.arange(M, device='cuda').expand(len(limits), M)
    # Compare each limit with the index matrix and set values to 1 where condition is true
    mat[valid_indices] = (index_matrix < limits.unsqueeze(1)).long()

    # Set entire rows to 1 where the value is -1
    mat[arr == -1] = 1

    # Return the matrix with added dimension to match the original output shape
    return torch.unsqueeze(mat.permute(1, 0), dim=2)
"""


@torch.jit.script
def first_false_index_gpu(arr: torch.Tensor):
    """
    For each column in arr, find the first index where the value is False.
    Args:
    - arr (numpy.ndarray): An array of shape (M,N) of booleans
    Returns:
    - numpy.ndarray: An array of shape (N,) containing the first index where the value is False in each column of arr.
                     If no False value is found in a column, the value is set to -1 for that column.
    """

    # Find where the tensor is False
    false_tensor = torch.zeros_like(arr, dtype=torch.bool)

    false_indices = torch.nonzero(arr == false_tensor)
    row_indices = false_indices[:, 0]
    col_indices = false_indices[:, 1]

    # Create a tensor of -1's to initialize the result
    result = torch.full((arr.shape[1],), -1, dtype=torch.int64, device=arr.device)

    # For each column index where we found a False value
    unique_cols = torch.unique(col_indices)
    for col in unique_cols:
        # Find the minimum row index for that column where the value is False
        result[col] = torch.min(row_indices[col_indices == col])
    return result


@torch.jit.script
def t_delete(tensor, indices):
    keep_mask = torch.ones(tensor.shape[1], dtype=torch.bool)
    keep_mask[indices] = False
    return tensor[:, keep_mask]


# @torch.jit.script
def batched_filter_gpu(
    path_matrix: torch.Tensor,
    dumped_values: torch.Tensor,
    i: int,
    dimensions: torch.Tensor,
    M: int,
    path_filter: torch.Tensor,
    current: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, N, _ = path_matrix.shape
    # print(f"Current status: \n path_matrix: {path_matrix.shape} \n dumped_values: {dumped_values.shape} \n path_filter: {path_filter.shape}")
    # Next, generate the "inside box" matrix operator and apply it
    # print("filtering by inside box: 1")
    inside_box_mat = Inside_Box_gpu(path_matrix[: i + 1, ...], dimensions)
    # print(inside_box_mat.shape)
    # print("filtering by inside box: 2")
    first_false = first_false_index_gpu(inside_box_mat)
    # print(len(first_false[first_false==-1]))
    # print("filtering by inside box: 3")
    # outside_box_filter = torch.tensor(generate_path_filter(first_false,M)).cuda()
    outside_box_filter = generate_path_filter_gpu(first_false, M + 2)
    torch.cuda.empty_cache()
    # print(outside_box_filter.shape)
    # print("filtering by inside box: 4")
    diff_matrix_box = path_matrix * outside_box_filter - path_matrix

    # box_indices = np.where(np.any(diff_matrix_box.cpu().numpy() != 0, axis=(0, 2)))[0]
    box_indices = torch.where(torch.any(torch.any(diff_matrix_box != 0, dim=0), dim=1))[
        0
    ]

    del diff_matrix_box
    ignore_indices = []
    box_stopping_points = torch.sum(outside_box_filter, dim=(0, 2)) - 1
    for n in box_indices:
        # Extract the first 3 and last 3 rows for column n
        idx = int(box_stopping_points[n])
        # print(idx)
        if idx >= i - 2:
            ignore_indices.append(n)
            continue
        new_data = (
            torch.cat((path_matrix[:3, n, :], path_matrix[idx : idx + 3, n, :]), dim=0)
            if idx != len(path_matrix) - 2
            else torch.cat((path_matrix[:3, n, :], path_matrix[-3:, n, :]), dim=0)
        )
        new_data = new_data.unsqueeze(
            1
        )  # Add a second dimension to match dumped_values shape
        # Concatenate new_data to dumped_values
        dumped_values = torch.cat(
            (dumped_values, new_data), dim=1
        )  # Concatenate along the second dimension
    # print("2.", dumped_values.shape)

    # First, get new path_matrix by multiplying the maximum path length randomly generated for each streamline
    # print("filtering by maximum path length")
    diff_matrix_path = (
        path_matrix * path_filter - path_matrix
    )  # Elementwise multiplication to zero values

    # path_indices = torch.tensor(np.where(np.any(diff_matrix_path.cpu().numpy() != 0, axis=(0, 2)))[0], device=diff_matrix_path.device)
    # path_indices = torch.where(torch.any(diff_matrix_path != 0, dim=(0, 2)))[0]
    # print(path_indices.shape)
    # path_indices = torch.nonzero(torch.any(diff_matrix_path != 0, dim=(0,2),keepdim=False))[:, 0]
    path_indices = torch.any(diff_matrix_path != 0, dim=(0, 2)).nonzero()[:, 0]

    del diff_matrix_path
    #print("FILTERING BY PATH INDICES")
    path_stopping_points = torch.sum(path_filter, dim=(0, 2)) - 1
    for n in path_indices:
        # Extract the first 3 and last 3 rows for column n
        if torch.any(box_indices == n):
            continue
        # print(idx)
        idx = int(path_stopping_points[n])
        if idx >= i - 2:
            ignore_indices.append(n)
            continue
        # print(idx)
        new_data = (
            torch.cat((path_matrix[:3, n, :], path_matrix[idx : idx + 3, n, :]), dim=0)
            if idx != len(path_matrix) - 2
            else torch.cat((path_matrix[:3, n, :], path_matrix[-3:, n, :]), dim=0)
        )
        new_data = new_data.unsqueeze(
            1
        )  # Add a second dimension to match dumped_values shape
        # print(path_matrix.shape)
        # print(new_data.shape)
        # print(dumped_values.shape)
        # Concatenate new_data to dumped_values
        dumped_values = torch.cat(
            (dumped_values, new_data), dim=1
        )  # Concatenate along the second dimension
    # print(dumped_values.shape)
    torch.cuda.empty_cache()
    # filter_indices = np.unique(np.concatenate((path_indices,box_indices)))
    filter_indices = torch.unique(torch.concatenate((path_indices, box_indices)))
    # Find elements in filter_indices that are not in ignore_indices
    mask = ~torch.isin(filter_indices, torch.tensor(ignore_indices).cuda())

    # Apply the mask to get the new filtered indices
    new_filter_indices = filter_indices[mask]

    # print(f"3. Amount of streamlines filtered: {len(path_indices)}, {len(box_indices), len(filter_indices)}")
    # path_mat = np.delete(path_matrix.cpu().numpy(), filter_indices, axis=1)
    # path_filt = np.delete(path_filter.cpu().numpy(), filter_indices, axis=1)
    # path_mat = torch.cat([path_matrix[:, :idx], path_matrix[:, idx+1:]], dim=1)
    # path_filt = torch.cat([path_filter[:, :idx], path_filter[:, idx+1:]], dim=1)
    path_mat = t_delete(path_matrix, new_filter_indices)
    path_filt = t_delete(path_filter, new_filter_indices)
    # print("4.", path_mat.shape)

    # return torch.tensor(path_mat).cuda(), torch.tensor(dumped_values).cuda(), torch.tensor(path_filt).cuda()
    return path_mat, dumped_values, path_filt


def batched_filter_gpu_end(
    path_matrix: torch.Tensor,
    dumped_values: torch.Tensor,
    i: int,
    dimensions: torch.Tensor,
    M: int,
    path_filter: torch.Tensor,
    current: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, N, _ = path_matrix.shape
    print(
        f"Current status: \n path_matrix: {path_matrix.shape} \n dumped_values: {dumped_values.shape} \n path_filter: {path_filter.shape}"
    )
    # Next, generate the "inside box" matrix operator and apply it
    # print("filtering by inside box: 1")
    inside_box_mat = Inside_Box_gpu(path_matrix[: i + 1, ...], dimensions)
    # print(inside_box_mat.shape)
    # print("filtering by inside box: 2")
    first_false = first_false_index_gpu(inside_box_mat)
    # print(len(first_false[first_false==-1]))
    # print("filtering by inside box: 3")
    # outside_box_filter = torch.tensor(generate_path_filter(first_false,M)).cuda()
    outside_box_filter = generate_path_filter_gpu(first_false, M + 2)
    torch.cuda.empty_cache()
    # print(outside_box_filter.shape)
    # print("filtering by inside box: 4")
    diff_matrix_box = path_matrix * outside_box_filter - path_matrix

    # box_indices = np.where(np.any(diff_matrix_box.cpu().numpy() != 0, axis=(0, 2)))[0]
    box_indices = torch.where(torch.any(torch.any(diff_matrix_box != 0, dim=0), dim=1))[
        0
    ]

    del diff_matrix_box

    box_stopping_points = torch.sum(outside_box_filter, dim=(0, 2)) - 1
    for n in box_indices:
        # Extract the first 3 and last 3 rows for column n
        idx = int(box_stopping_points[n])
        new_data = (
            torch.cat((path_matrix[:3, n, :], path_matrix[idx : idx + 3, n, :]), dim=0)
            if idx != len(path_matrix) - 2
            else torch.cat((path_matrix[:3, n, :], path_matrix[-3:, n, :]), dim=0)
        )
        new_data = new_data.unsqueeze(
            1
        )  # Add a second dimension to match dumped_values shape
        # Concatenate new_data to dumped_values
        dumped_values = torch.cat(
            (dumped_values, new_data), dim=1
        )  # Concatenate along the second dimension
    # print("2.", dumped_values.shape)

    # First, get new path_matrix by multiplying the maximum path length randomly generated for each streamline
    # print("filtering by maximum path length")
    diff_matrix_path = (
        path_matrix * path_filter - path_matrix
    )  # Elementwise multiplication to zero values

    # path_indices = torch.tensor(np.where(np.any(diff_matrix_path.cpu().numpy() != 0, axis=(0, 2)))[0], device=diff_matrix_path.device)
    # path_indices = torch.where(torch.any(diff_matrix_path != 0, dim=(0, 2)))[0]
    # print(path_indices.shape)
    # path_indices = torch.nonzero(torch.any(diff_matrix_path != 0, dim=(0,2),keepdim=False))[:, 0]
    path_indices = torch.any(diff_matrix_path != 0, dim=(0, 2)).nonzero()[:, 0]

    del diff_matrix_path

    path_stopping_points = torch.sum(path_filter, dim=(0, 2)) - 1
    for n in path_indices:
        # Extract the first 3 and last 3 rows for column n
        if torch.any(box_indices == n):
            continue
        idx = int(path_stopping_points[n])
        # print(idx)
        new_data = (
            torch.cat((path_matrix[:3, n, :], path_matrix[idx : idx + 3, n, :]), dim=0)
            if idx != len(path_matrix) - 2
            else torch.cat((path_matrix[:3, n, :], path_matrix[-3:, n, :]), dim=0)
        )
        new_data = new_data.unsqueeze(
            1
        )  # Add a second dimension to match dumped_values shape
        # print(path_matrix.shape)
        # print(new_data.shape)
        # print(dumped_values.shape)
        # Concatenate new_data to dumped_values
        dumped_values = torch.cat(
            (dumped_values, new_data), dim=1
        )  # Concatenate along the second dimension
    # print(dumped_values.shape)
    torch.cuda.empty_cache()
    # filter_indices = np.unique(np.concatenate((path_indices,box_indices)))
    filter_indices = torch.unique(torch.concatenate((path_indices, box_indices)))
    # print(f"3. Amount of streamlines filtered: {len(path_indices)}, {len(box_indices), len(filter_indices)}")
    # path_mat = np.delete(path_matrix.cpu().numpy(), filter_indices, axis=1)
    # path_filt = np.delete(path_filter.cpu().numpy(), filter_indices, axis=1)
    # path_mat = torch.cat([path_matrix[:, :idx], path_matrix[:, idx+1:]], dim=1)
    # path_filt = torch.cat([path_filter[:, :idx], path_filter[:, idx+1:]], dim=1)
    path_mat = t_delete(path_matrix, filter_indices)
    path_filt = t_delete(path_filter, filter_indices)
    # print("4.", path_mat.shape)

    # return torch.tensor(path_mat).cuda(), torch.tensor(dumped_values).cuda(), torch.tensor(path_filt).cuda()
    return path_mat, dumped_values, path_filt
