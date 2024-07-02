import numpy as np
import pandas as pd
from numba import jit
import time
import torch
from CPET.utils.io import parse_pqr

#@profile
def calculate_electric_field_torch(x_0, x, Q):
    """
    Computes electric field at multiple points given positions of charges
    Takes
        x_0(tensor) - positions to compute field at of shape (N,3)
        x(tensor) - positions of charges of shape (L,3)
        Q(tensor) - magnitude and sign of charges of shape (L,1)
    Returns
        E(tensor) - electric field at the points of shape (N,3)
    """
    # Compute the difference between every point in x_0 and every point in x
    # R will be of shape (N, L, 3)
    R = x_0.unsqueeze(1)-x.unsqueeze(0)
    r_mag = torch.norm(R, dim=-1)
    r_mag_cube = (r_mag**3).unsqueeze(-1)

    # Reshape Q for proper broadcasting
    Q_reshaped = Q.unsqueeze(0)

    # Sum over the L charges to get the net electric field at each point in x_0
    E = torch.einsum("ijk,ijk,ijk->ik", R, 1 / r_mag_cube, Q_reshaped) * 14.3996451
    return E

#@profile
def propagate_topo_matrix(path_matrix,i, x, Q, step_size):
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
    path_matrix_prior = path_matrix[i]
    E = calculate_electric_field_torch(path_matrix_prior, x, Q)  # Compute field
    E_norm = torch.norm(E, dim=-1).unsqueeze(-1)
    E_normalized = E / E_norm
    path_matrix[i+1] = path_matrix_prior + step_size*E_normalized
    return path_matrix

def curv_mat(v_prime, v_prime_prime):
    """
    Computes curvature of the streamline at a given point
    Takes
        v_prime(array) - first derivative of streamline at the point of shape (N,3)
        v_prime_prime(array) - second derivative of streamline at the point of shape (N,3)
    Returns
        curvature(float) - the curvature
    """
    curvature = torch.norm(torch.cross(v_prime, v_prime_prime), dim=-1) / torch.norm(v_prime, dim=-1) ** 3
    return curvature

def compute_curv_and_dist_mat(x_init,x_init_plus,x_init_plus_plus,x_0,x_0_plus,x_0_plus_plus):
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
    curv_init = curv_mat(x_init_plus - x_init, x_init_plus_plus - 2 * x_init_plus + x_init)
    curv_final = curv_mat(x_0_plus - x_0, x_0_plus_plus - 2 * x_0_plus + x_0)
    curv_mean = (curv_init + curv_final) / 2
    dist = torch.norm(x_init - x_0, dim=-1)
    return dist, curv_mean
    
def Inside_Box(local_points, dimensions):
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
        (local_points[..., 0] >= -half_length) & (local_points[..., 0] <= half_length) &
        (local_points[..., 1] >= -half_width) & (local_points[..., 1] <= half_width) &
        (local_points[..., 2] >= -half_height) & (local_points[..., 2] <= half_height)
    )
    return is_inside

def initialize_streamline_grid(center, x, y, dimensions, n_samples, step_size):
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
    random_max_samples = np.random.randint(1, M, N)
    path_matrix = np.zeros((M+2,N,3))
    path_matrix[0] = random_points_local
    path_filter = generate_path_filter(random_max_samples,M)
    print(M,N)
    return path_matrix, transformation_matrix, M, path_filter,random_max_samples

def generate_path_filter(arr, M):
    # Initialize the matrix with zeros
    mat = np.zeros((len(arr), M+2), dtype=int)

    # Iterate over the array
    for i, value in enumerate(arr):
        # Set the values to 1 up to and including 2 after the entry value
        mat[i, :value+2] = 1
    return np.expand_dims(mat.T,axis=2)

def first_false_index(arr):
    """
    For each column in arr, find the first index where the value is False.
    Args:
    - arr (numpy.ndarray): An array of shape (M, N, 1) of booleans
    Returns:
    - numpy.ndarray: An array of shape (N,) containing the first index where the value is False in each column of arr. 
                     If no False value is found in a column, the value is set to -1 for that column.
    """
    
    # Find where the tensor is False
    false_indices = torch.nonzero(arr == False, as_tuple=True)
    row_indices, col_indices = false_indices

    # Create a tensor of -1's to initialize the result
    result = torch.full((arr.shape[1],), -1, dtype=torch.int64)

    # For each column index where we found a False value
    unique_cols = torch.unique(col_indices)
    for col in unique_cols:
        # Find the minimum row index for that column where the value is False
        min_row_for_col = torch.min(row_indices[col_indices == col])
        result[col] = min_row_for_col 
    return result

def filter_Inside_Box(path_matrix, dimensions,M,path_filter):
    _, N, _ = path_matrix.shape
    #First, get new path_matrix by multiplying the maximum path length randomly generated for each streamline
    print("filtering by maximum path length")
    filtered_path_matrix = path_matrix * path_filter #Elementwise multiplication to zero values

    #Next, generate the "inside box" matrix operator and apply it
    print("filtering by inside box: 1")
    inside_box_mat = Inside_Box(path_matrix, dimensions)
    print("filtering by inside box: 2")
    first_false = first_false_index(inside_box_mat).cpu().numpy()
    print("filtering by inside box: 3")
    outside_box_filter = generate_path_filter(first_false,M)
    print("filtering by inside box: 4")
    filtered_path_matrix = filtered_path_matrix.cpu().numpy()*outside_box_filter

    #Find last 3 non-zero values for each row
    final_mat = np.zeros((3, N, 3))

    for j in range(N):
        non_zero_indices = np.where(np.any(filtered_path_matrix[:, j, :] != [0, 0, 0], axis=-1))[0]
        # If there are fewer than 3 non-zero elements, handle that case
        last_three_indices = non_zero_indices[-3:] if len(non_zero_indices) >= 3 else non_zero_indices
        final_mat[:len(last_three_indices), j, :] = filtered_path_matrix[last_three_indices, j, :]

    return final_mat, filtered_path_matrix

#@profile
def main():
    options = {
        "path_to_pqr": "./1_wt_run1_0.pqr",
        "center": [55.965, 46.219, 22.123],
        "x": [56.191, 48.344, 22.221],
        "y": [57.118, 46.793, 20.46],
        "n_samples": 10000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.01,
    }
    x, Q = parse_pqr(options["path_to_pqr"])
    Q=torch.tensor(Q).cuda()
    center = np.array(options["center"])
    x_vec_pt = np.array(options["x"])
    y_vec_pt = np.array(options["y"])
    dimensions = np.array(options["dimensions"])
    step_size = options["step_size"]
    n_samples = options["n_samples"]
    path_matrix, transformation_matrix, M, path_filter, random_max_samples = initialize_streamline_grid(center, x_vec_pt, y_vec_pt, dimensions, n_samples, step_size)
    path_matrix_torch=torch.tensor(path_matrix).cuda()
    path_filter=torch.tensor(path_filter).cuda()
    x = (x-center)@np.linalg.inv(transformation_matrix)
    x=torch.tensor(x).cuda()
    j=0
    start_time = time.time()
    for i in range(len(path_matrix)):
      print(j)
      if j == len(path_matrix)-1:
        break
      path_matrix_torch = propagate_topo_matrix(path_matrix_torch,i, x, Q, step_size)
      j+=1
    init_matrix=path_matrix_torch[0:3,:,:]
    final_matrix,_=filter_Inside_Box(path_matrix_torch,dimensions,M,path_filter)
    final_matrix = torch.tensor(final_matrix).cuda()
    distances, curvatures = compute_curv_and_dist_mat(init_matrix[0,:,:], init_matrix[1,:,:], init_matrix[2,:,:],final_matrix[0,:,:],final_matrix[1,:,:],final_matrix[2,:,:])
    end_time = time.time()
    print(f"Time taken for {options['n_samples']} calculations with N~4000: {end_time - start_time:.2f} seconds")
    topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
    np.savetxt("hist_cpet_mat.txt", topology)
    return topology

main()
'''x = np.array([[0,0,0]])
x_0 = np.array([[0,0,0.5],[0,0,1],[0.5,0,0]])
Q = np.array([[1]])
print(calculate_electric_field(x_0,x,Q))'''

