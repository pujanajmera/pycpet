import time
import torch
from typing import Tuple
from torch.profiler import profile

"""
Universal variable notation for GPU streamline computation:
L: The number of streamlines
M: The number of points within a single streamline. This can
be a full streamline (like the original path matrix), or a batch size
N: The number of charges
F: Number of dumped values during filtering
P(_K): function-to-function placeholder parameter
"""


def check_tensor(x, name="Tensor"):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"{name} contains NaN or Inf")
        return True
    return False


# @profile
@torch.jit.script
def calculate_electric_field_torch_batch_gpu(
    x_0: torch.Tensor, x: torch.Tensor, Q: torch.Tensor
) -> torch.Tensor:
    """
    Computes field at a set of points given the charges and their positions
    Takes
        x_0(array) - positions to compute field at of shape (L,3)
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
    Returns
        E - electric field at x_0 of shape (L,3)
    """
    N = x_0.size(0)
    E = torch.zeros(N, 3, device=x_0.device, dtype=torch.float32)

    for start in range(0, N, 100):
        end = min(start + 100, N)
        # print(x_0[start:end].unsqueeze(1).shape)
        # print(x.unsqueeze(0).shape)
        R = x_0[start:end].unsqueeze(1) - x.unsqueeze(0)
        r_mag_cube = torch.norm(R, dim=-1, keepdim=True).pow(-3)
        # E[start:end] = torch.einsum("ijk,ijk,ijk->ik", R, 1/r_mag_cube, Q) * 14.3996451
        E[start:end] = (R * r_mag_cube * Q).sum(dim=1) * 14.3996451

    return E


# @torch.jit.script
def propagate_topo_matrix_gpu(
    path_matrix: torch.Tensor,
    i: torch.Tensor,
    x: torch.Tensor,
    Q: torch.Tensor,
    step_size: torch.Tensor,
    # dtype_str: str
) -> torch.Tensor:
    """
    Propagates position based on normalized electric field at a set of points
    Takes
        path_matrix(array) - positions of streamline of shape (M,L,3)
        i(int) - most recently updated position in streamline
        x(array) - positions of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
        step_size(float) - size of streamline step to take when propagating, real and positive
    Returns
        x_0 - new position on streamline after propagation via electric field
    """
    path_matrix_prior = path_matrix[int(i)]
    N = path_matrix_prior.size(0)
    """
    if dtype_str == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float64
    """
    E = calculate_electric_field_torch_batch_gpu(path_matrix_prior, x, Q)
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
        local_point(array) - current local points of shape (M,L,3)
        dimensions(array) - L, W, H of box of shape (1,3)
    Returns
        is_inside(bool) - whether the point is inside the box or shape (M,L)
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


# @torch.jit.script
def generate_path_filter_gpu(arr, M):
    """
    Generates a path filter based on booleans, either from path limit or box edge
    Takes
        arr(array) - array of shape (L,) of streamline end values as indices. Non-end streamlines are indicated by -1s
        M(int) - number of points in streamline. For path filter initialization, M=M+2 points are called since
        that is the matrix size
    Returns
        mat(array) - path filter of shape (M,L,1) that has 1's everywhere that you want to keep info
        (e.g. the streamline has not ended its path + 2 or the streamline has not hit the box edge + 2)
    """
    # Initialize the matrix with zeros

    mat = torch.zeros(
        (len(arr), int(M)), dtype=torch.int64, device="cuda"
    )  # Shape (L,M)

    # Iterate over the array
    for i, value in enumerate(arr):  # Enumerate is effectively shape (L,2)
        # If a streamline end exists, set the row to 1 up to the end point +2 additional points to ensure first and second derivatives
        # Since value is the index, you need to add 3 to make sure you get the point and 2 additional points
        # For cases where value+2 is greater than the max number of streamline points, the entire streamline would be set to 1
        if value != -1:
            mat[i, :value] = 1
        # If no streamline ends exist, set the entire row to 1 (meaning nothing to filter)
        else:
            mat[i] = 1
    # return np.expand_dims(mat.T,axis=2)
    return torch.unsqueeze(mat.permute(1, 0), dim=2)


# @torch.jit.script
def first_false_index_gpu(arr: torch.Tensor):
    """
    For each streamline, find the first place where the streamline hits the box edge
    Args:
    - arr (numpy.ndarray): An array of shape (M,L) of booleans
    Returns:
    - numpy.ndarray: An array of shape (L,) containing, for each streamline (column), the first place (row) where it is outside of the box.
                     If no False value is found in a streamline (column), the value is set to -1 for that streamline (column).
    """

    # Find where the tensor is False
    false_tensor = torch.zeros_like(arr, dtype=torch.bool)  # Shape (M,L)

    false_indices = torch.nonzero(
        arr == false_tensor
    )  # Shape (P_0,2), where P_0 is the number of False vals in the whole matrix
    row_indices = false_indices[:, 0]  # Shape (P_0,)
    col_indices = false_indices[:, 1]  # Shape (P_0,)

    # Create a tensor of -1's to initialize the result
    result = torch.full(
        (arr.shape[1],), -1, dtype=torch.int64, device=arr.device
    )  # Shape (L,)

    # For each column index where we found a False value
    unique_cols = torch.unique(
        col_indices
    )  # Shape (P_1,), where P_1 is the number of streamlines that have at least one point outside the box
    for col in unique_cols:
        # Find the first place along the streamline where the point is outside the box
        result[col] = torch.min(row_indices[col_indices == col])
    return result


@torch.jit.script
def t_delete(tensor, indices):
    keep_mask = torch.ones(tensor.shape[1], dtype=torch.bool)
    keep_mask[indices] = False
    return tensor[:, keep_mask]


def filter_and_dump_gpu(
    GPU_batch_freq,
    stopping_points,
    stopping_indices,
    init_points,
    path_matrix,
    dumped_values,
):
    """
    Uses either box or path indices to dump values from path matrix or
    identify indices to ignore
    Takes
        GPU_batch_freq(int) - frequency of GPU batch, typically kept at 100
        stopping_points(array) - stopping points of streamline based on box or path
        stopping_indices(array) - indices of stopping points
        init_points(array) - initial points of all current streamlines of shape (3,L,3)
        path_matrix(array) - positions of streamline of shape (M,L,3)
        dumped_values(array) - values to dump
    Returns
        dumped_values(array) - updated dumped streamline values
        ignore_indices(array) - indices to ignore
    """
    ignore_indices = []
    for i in range(len(stopping_indices)):
        idx = int(stopping_points[i])
        if idx >= GPU_batch_freq - 2:
            ignore_indices.append(stopping_indices[i])
            continue
        new_data = torch.concat(
            (
                init_points[:, stopping_indices[i], :],
                path_matrix[idx : idx + 3, stopping_indices[i], :],
            ),
            dim=0,
        )
        dumped_values = torch.cat((dumped_values, new_data.unsqueeze(1)), dim=1)
        # print(f"Final {filtertype} point from dumped_values: {dumped_values[:, -1, :]}")
    print(
        f"Now have {dumped_values.shape[1]} number of dumped streamlines after filtering."
    )
    return dumped_values, ignore_indices


# @torch.jit.script
def batched_filter_gpu(
    path_matrix: torch.Tensor,
    dumped_values: torch.Tensor,
    i: int,
    dimensions: torch.Tensor,
    path_filter: torch.Tensor,
    GPU_batch_freq,
    init_points,
    dtype_str: str = "float64",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch filtering of the path matrix, and dumping streamline values
    that hit their limits, and reinitalizing path matrix and filters
    Takes
        path_matrix(array) - positions of streamline of shape (GPU_batch_filter,L,3)
        dumped_values(array) - previous streamline begin+end of shape (F0,6)
        i(int) - current iteration
        dimensions(array) - L, W, H of box of shape (1,3)
        path_filter(array) - filter to apply to path matrix of shape (M,L,1)
        GPU_batch_freq(int) - frequency of GPU batch, changes if remainder exists
        init_points(array) - initial points of all current streamlines of shape (3,L,3)
        dtype_str(str) - data type of path matrix
    Returns
        path_mat_new(array) - new path matrix of shape (M,L-F1,3) retaining last two values
        dumped_values(array) - updated streamline begin+end of shape (F0+F1,6)
        path_filt(array) - new path filter of shape of shape (L-F1)
        init_points_new(array) - new initial points of shape (3,L-F1,3)
    """
    print("Filtering...")
    inside_box_mat = Inside_Box_gpu(path_matrix, dimensions)  # Shape (M,L)
    first_false = first_false_index_gpu(inside_box_mat)  # Shape (L,)
    outside_box_filter = generate_path_filter_gpu(
        first_false, GPU_batch_freq
    )  # Shape (M,L,1)
    torch.cuda.empty_cache()
    diff_matrix_box = path_matrix * outside_box_filter - path_matrix  # Shape (M,L,3)
    box_indices = torch.where(torch.any(torch.any(diff_matrix_box != 0, dim=0), dim=1))[
        0
    ]  # Shape (F1,) --> which streamlines hit the box edge

    box_stopping_points = torch.sum(outside_box_filter, dim=(0, 2))  # Shape (L,)

    del diff_matrix_box

    path_filter_temp = path_filter[
        i * (GPU_batch_freq - 2) : GPU_batch_freq + i * (GPU_batch_freq - 2), ...
    ]  # Shape (GPU_batch_freq,L,1)
    diff_matrix_path = (
        path_matrix * path_filter_temp - path_matrix
    )  # Shape (GPU_batch_freq,L,3)

    path_indices = torch.any(diff_matrix_path != 0, dim=(0, 2)).nonzero()[:, 0]

    del diff_matrix_path
    path_stopping_points = torch.sum(path_filter_temp, dim=(0, 2))

    filter_indices = torch.unique(torch.concatenate((path_indices, box_indices)))

    stopping_points = []

    for i, index in enumerate(filter_indices):
        if index in path_indices and index in box_indices:

            stopping_points.append(
                min(
                    path_stopping_points[index],
                    box_stopping_points[index],
                )
            )

            # Slower version for testing:
            """
            print(f"Streamline detected to hit both path and box at index: {index}, checking now...")
            if path_stopping_points[index] < box_stopping_points[index]:
                stopping_points.append(path_stopping_points[index])
                print("Streamline hit path limit first at index", index)
            elif path_stopping_points[index] > box_stopping_points[index]:
                stopping_points.append(box_stopping_points[index])
                print("Streamline hit box first at index", index)
            else:
                stopping_points.append(path_stopping_points[index])
                print("Path and box indices are equal")
            """

        elif index in path_indices:
            stopping_points.append(path_stopping_points[index])
        elif index in box_indices:
            stopping_points.append(first_false[index])
        else:
            raise ValueError(
                "Index not found in either path or box indices for some reason..."
            )

    dumped_values, ignore_indices = filter_and_dump_gpu(
        GPU_batch_freq,
        stopping_points,
        filter_indices,
        init_points,
        path_matrix,
        dumped_values,
    )

    torch.cuda.empty_cache()

    mask = ~torch.isin(filter_indices, torch.tensor(ignore_indices).cuda())

    # Apply the mask to get the new filtered indices
    new_filter_indices = filter_indices[mask]

    path_mat = t_delete(path_matrix, new_filter_indices)
    path_filt = t_delete(path_filter, new_filter_indices)
    init_points_new = t_delete(init_points, new_filter_indices)
    # print("4.", path_mat.shape)
    if dtype_str == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float64
    path_mat_new = torch.zeros(
        GPU_batch_freq, path_mat.shape[1], 3, device=path_mat.device, dtype=dtype
    )
    path_mat_new[0:2, ...] = path_mat[-2:, ...]
    print(f"Number of streamlines filtered: {len(new_filter_indices)}")

    # return torch.tensor(path_mat).cuda(), torch.tensor(dumped_values).cuda(), torch.tensor(path_filt).cuda()
    return path_mat_new, dumped_values, path_filt, init_points_new
