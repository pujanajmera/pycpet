import numpy as np
import numba as nb
from numba import jit
import time
from CPET.utils.fastmath import power, nb_subtract, custom_norm_3d

#@profile

@nb.njit(parallel=True)
def einsum(R, r_mag_cube, Q_reshaped):
    # Define a small epsilon to avoid division by zero
    epsilon = 1e-10

    # Resulting array with appropriate dimensions
    E_batch = np.zeros((R.shape[0], R.shape[2]), dtype=np.float32)

    # Nested loops to perform the operation equivalent to the einsum
    for i in nb.prange(R.shape[0]):  # Loop over the first dimension
        for k in nb.prange(R.shape[2]):  # Loop over the third dimension
            sum_product = 0.0
            for j in nb.prange(R.shape[1]):  # Summing over the second dimension
                # Add epsilon in the denominator to avoid division by zero
                safe_denominator = r_mag_cube[i, j, k] + epsilon
                sum_product += R[i, j, k] * (1 / safe_denominator) * Q_reshaped[0, j, k]
            E_batch[i, k] = sum_product * 14.3996451
    return E_batch

def calculate_electric_field_grid_gridcpu(x_0, x, Q):
    N = len(x_0)
    E = np.zeros((N, 3))
    Q_reshaped=np.expand_dims(Q, axis=0)
    for start in range(0, N, 10):
        end = min(start + 10, N)
        x_0_batch = x_0[start:end]
        #start_time = time.time()
        R = nb_subtract(np.expand_dims(x_0_batch, axis=1), np.expand_dims(x, axis=0))
        #print(f"Time to subtract: {time.time()-start_time}")
        #start_time = time.time()
        #r_mag = np.linalg.norm(R, axis=-1, keepdims=True)
        r_mag = custom_norm_3d(R)
        #print(f"Time to calculate norm: {time.time()-start_time}")
        #print(r_mag.shape)
        #start_time = time.time()
        r_mag_cube = power(r_mag, 3)
        #print(f"Time to calculate power: {time.time()-start_time}")
        #start_time = time.time()
        #E_batch = np.einsum("ijk,ijk,ijk->ik", R, 1 / r_mag_cube, Q_reshaped) * 14.3996451
        E_batch = einsum(R, r_mag_cube, Q_reshaped)
        #print(f"Time to calculate E: {time.time()-start_time}")
        E[start:end] = E_batch
    return E

def propagate_topo_matrix_gridcpu(path_matrix, i, x, Q, step_size):
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
    E = calculate_electric_field_grid_gridcpu(path_matrix_prior, x, Q)  # Compute field
    E_norm = np.expand_dims(np.linalg.norm(E, axis=-1),axis=-1)
    E_normalized = E / E_norm
    path_matrix[i+1] = path_matrix_prior + step_size*E_normalized
    return path_matrix

def curv_mat_gridcpu(v_prime, v_prime_prime):
    """
    Computes curvature of the streamline at a given point
    Takes
        v_prime(array) - first derivative of streamline at the point of shape (N,3)
        v_prime_prime(array) - second derivative of streamline at the point of shape (N,3)
    Returns
        curvature(float) - the curvature
    """
    curvature = np.linalg.norm(np.cross(v_prime, v_prime_prime), axis=-1) / np.linalg.norm(v_prime, axis=-1) ** 3
    return curvature


def compute_curv_and_dist_mat_gridcpu(x_init,x_init_plus,x_init_plus_plus,x_0,x_0_plus,x_0_plus_plus):
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
    curv_init = curv_mat_gridcpu(x_init_plus - x_init, x_init_plus_plus - 2 * x_init_plus + x_init)
    curv_final = curv_mat_gridcpu(x_0_plus - x_0, x_0_plus_plus - 2 * x_0_plus + x_0)
    curv_mean = (curv_init + curv_final) / 2
    dist = np.linalg.norm(x_init - x_0, axis=-1)
    return dist, curv_mean
    
#@nb.njit
def Inside_Box_gridcpu(local_points, dimensions):
    """
    Checks if a streamline point is inside a box
    Takes
        local_point(array) - current local point of shape (1,3)
        dimensions(array) - L, W, H of box of shape (1,3)
    Returns
        is_inside(bool) - whether the point is inside the box
    """
    # Convert lists to numpy arrays
    half_length, half_width, half_height = dimensions[0],dimensions[1],dimensions[2]
    # Check if the point lies within the dimensions of the box
    is_inside = (
        (local_points[..., 0] >= -half_length) & (local_points[..., 0] <= half_length) &
        (local_points[..., 1] >= -half_width) & (local_points[..., 1] <= half_width) &
        (local_points[..., 2] >= -half_height) & (local_points[..., 2] <= half_height)
    )
    return is_inside


def initialize_streamline_grid_gridcpu(center, x, y, dimensions, n_samples, step_size):
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
    N=n_samples
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
    random_x = np.random.uniform(-half_length, half_length, N)
    random_y = np.random.uniform(-half_width, half_width, N)
    random_z = np.random.uniform(-half_height, half_height, N)
    # Each row in random_points_local corresponds to x, y, and z coordinates of a point in the box's coordinate system
    random_points_local = np.column_stack([random_x, random_y, random_z])
    # Convert these points back to the global coordinate system
    transformation_matrix = np.column_stack(
        [x_unit, y_unit, z_unit]
    ).T  # Each column is a unit vector
    max_distance = 2 * np.linalg.norm(
        np.array(dimensions)
    )  # Define maximum sample limit as 2 times the diagonal
    M = round(max_distance/step_size)
    random_max_samples = np.random.randint(1, M, N, dtype=np.int32)
    path_matrix = np.zeros((M+2,N,3),dtype=np.float32)
    path_matrix[0] = random_points_local
    path_filter = generate_path_filter_gridcpu(random_max_samples,M)
    print(M, N)
    return path_matrix, transformation_matrix, M, path_filter,random_max_samples

def generate_path_filter_gridcpu(arr, M):
    # Initialize the matrix with zeros

    mat = np.zeros((len(arr), M+2))

    # Iterate over the array
    for i, value in enumerate(arr):
        # Set the values to 1 up to and including 2 after the entry value
        if value != -1:
            mat[i, :value+2] = 1
        else:
            mat[i] = 1
    #return np.expand_dims(mat.T,axis=2)
    return np.expand_dims(mat.T, axis=2)

def first_false_index_gridcpu(arr):
    """
    For each column in arr, find the first index where the value is False.
    Args:
    - arr (numpy.ndarray): An array of shape (M, N, 1) of booleans
    Returns:
    - numpy.ndarray: An array of shape (N,) containing the first index where the value is False in each column of arr. 
                     If no False value is found in a column, the value is set to -1 for that column.
    """
    
    # Find where the tensor is False
    false_indices = np.nonzero(arr == False)
    row_indices, col_indices = false_indices

    # Create a tensor of -1's to initialize the result
    result = np.full((arr.shape[1],), -1)

    # For each column index where we found a False value
    unique_cols = np.unique(col_indices)
    for col in unique_cols:
        # Find the minimum row index for that column where the value is False
        min_row_for_col = np.min(row_indices[col_indices == col])
        result[col] = min_row_for_col 
    return result

def t_delete(tensor, indices):
    keep_mask = np.ones(tensor.shape[1], dtype=bool)
    keep_mask[indices] = False
    new_tensor = tensor[:, keep_mask]
    return new_tensor

def batched_filter_gridcpu(path_matrix, dumped_values, i, dimensions, M, path_filter, current=True):
    _, N, _ = path_matrix.shape
    #print(f"Current status: \n path_matrix: {path_matrix.shape} \n dumped_values: {dumped_values.shape} \n path_filter: {path_filter.shape}")
    #First, get new path_matrix by multiplying the maximum path length randomly generated for each streamline
    #print("filtering by maximum path length")
    diff_matrix_path = path_matrix * path_filter - path_matrix#Elementwise multiplication to zero values
    path_indices = np.where(np.any(np.any(diff_matrix_path != 0, axis=0), axis=1))[0]
    path_stopping_points = np.sum(path_filter,axis=(0,2))-1
    for n in path_indices:
        # Extract the first 3 and last 3 rows for column n
        idx = int(path_stopping_points[n])
        if idx == len(path_matrix)-2:
            new_data = np.concatenate((path_matrix[:3, n, :], path_matrix[-3:, n, :]), axis=0)
        else:
            new_data = np.concatenate((path_matrix[:3, n, :], path_matrix[idx:idx+3, n, :]), axis=0)
        new_data = np.expand_dims(new_data, axis=1)  # Add a second dimension to match dumped_values shape
        #print(path_matrix.shape)
        #print(new_data.shape)
        #print(dumped_values.shape)
        # Concatenate new_data to dumped_values
        dumped_values = np.concatenate((dumped_values, new_data), axis=1)  # Concatenate along the second dimension
    #print(dumped_values.shape)
    #Next, generate the "inside box" matrix operator and apply it
    #print("filtering by inside box: 1")
    inside_box_mat = Inside_Box_gridcpu(path_matrix[:i+1,...], dimensions)
    #print(inside_box_mat.shape)
    #print("filtering by inside box: 2")
    first_false = first_false_index_gridcpu(inside_box_mat)
    #print(len(first_false[first_false==-1]))
    #print("filtering by inside box: 3")
    #outside_box_filter = torch.tensor(generate_path_filter(first_false,M)).cuda()
    outside_box_filter = generate_path_filter_gridcpu(first_false,M)
    
    #print(outside_box_filter.shape)
    #print("filtering by inside box: 4")
    diff_matrix_box = path_matrix * outside_box_filter - path_matrix
    #box_indices = np.where(np.any(diff_matrix_box.cpu().numpy() != 0, axis=(0, 2)))[0]
    box_indices = np.where(np.any(np.any(diff_matrix_box != 0, axis=0), axis=1))[0]

    box_stopping_points = np.sum(outside_box_filter,axis=(0,2))-1
    for n in box_indices:
        # Extract the first 3 and last 3 rows for column n
        if n in path_indices:
            continue
        idx = int(box_stopping_points[n])
        new_data = np.concatenate((path_matrix[:3, n, :], path_matrix[idx:idx+3, n, :]), axis=0)
        new_data = np.expand_dims(new_data, axis=1)  # Add a second dimension to match dumped_values shape
        # Concatenate new_data to dumped_values
        dumped_values = np.concatenate((dumped_values, new_data), axis=1)  # Concatenate along the second dimension
    #print("2.", dumped_values.shape)
    #filter_indices = np.unique(np.concatenate((path_indices,box_indices)))
    filter_indices = np.unique(np.concatenate((path_indices, box_indices)))
    #print(f"3. Amount of streamlines filtered: {len(path_indices)}, {len(box_indices), len(filter_indices)}")
    #path_mat = np.delete(path_matrix.cpu().numpy(), filter_indices, axis=1)
    #path_filt = np.delete(path_filter.cpu().numpy(), filter_indices, axis=1)
    #path_mat = torch.cat([path_matrix[:, :idx], path_matrix[:, idx+1:]], dim=1)
    #path_filt = torch.cat([path_filter[:, :idx], path_filter[:, idx+1:]], dim=1)
    path_mat = t_delete(path_matrix, filter_indices)
    path_filt = t_delete(path_filter, filter_indices)
    #print("4.", path_mat.shape)

    #return torch.tensor(path_mat).cuda(), torch.tensor(dumped_values).cuda(), torch.tensor(path_filt).cuda()
    return path_mat, dumped_values, path_filt




