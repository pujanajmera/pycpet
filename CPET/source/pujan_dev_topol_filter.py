# compute_topol_grid_activefilter.py


import numpy as np
import time
import torch
import torch.jit as jit
from CPET.utils.parser import parse_pqr

#@profile
@torch.jit.script
def calculate_electric_field_torch_batch(x_0: torch.Tensor, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    N = x_0.size(0)
    L = x.size(0)
    E = torch.zeros(N, 3, device=x_0.device)
    Q_reshaped=Q.unsqueeze(0)
    for start in range(0, N, 1000):
        end = min(start + 1000, N)
        x_0_batch = x_0[start:end]
        R = x_0_batch.unsqueeze(1) - x.unsqueeze(0)
        r_mag = torch.norm(R, dim=-1, keepdim=True)
        r_mag_cube = r_mag.pow(3)
        E_batch = torch.einsum("ijk,ijk,ijk->ik", R, 1 / r_mag_cube, Q_reshaped) * 14.3996451
        E[start:end] = E_batch
    return E

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
    E = calculate_electric_field_torch_batch(path_matrix_prior, x, Q)  # Compute field
    E_norm = torch.norm(E, dim=-1).unsqueeze(-1)
    E_normalized = E / E_norm
    path_matrix[i+1] = path_matrix_prior + step_size*E_normalized
    return path_matrix

@torch.jit.script
def curv_mat(v_prime: torch.Tensor, v_prime_prime: torch.Tensor) -> torch.Tensor:
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
    print(M, N)
    return path_matrix, transformation_matrix, M, path_filter,random_max_samples

def generate_path_filter(arr, M):
    # Initialize the matrix with zeros
    mat = torch.zeros((len(arr), M+2), dtype=int, device='cuda')

    # Iterate over the array
    for i, value in enumerate(arr):
        # Set the values to 1 up to and including 2 after the entry value
        if value != -1:
            mat[i, :value+2] = 1
        else:
            mat[i] = 1
    #return np.expand_dims(mat.T,axis=2)
    return torch.unsqueeze(mat.permute(1, 0), dim=2)

def batched_filter(path_matrix, dumped_values, i, dimensions, M, path_filter, current=True):
    _, N, _ = path_matrix.shape
    #print(f"Current status: \n path_matrix: {path_matrix.shape} \n dumped_values: {dumped_values.shape} \n path_filter: {path_filter.shape}")
    #First, get new path_matrix by multiplying the maximum path length randomly generated for each streamline
    #print("filtering by maximum path length")
    diff_matrix_path = path_matrix * path_filter - path_matrix#Elementwise multiplication to zero values
    #path_indices = np.where(np.any(diff_matrix_path.cpu().numpy() != 0, axis=(0, 2)))[0]
    #path_indices = torch.where(torch.any(diff_matrix_path != 0, dim=(0, 2)))[0]
    path_indices = torch.where(torch.any(torch.any(diff_matrix_path != 0, dim=0), dim=1))[0]

    path_stopping_points = torch.sum(path_filter,dim=(0,2))-1
    for n in path_indices:
        # Extract the first 3 and last 3 rows for column n
        idx = path_stopping_points[n]
        new_data = torch.cat((path_matrix[:3, n, :], path_matrix[idx:idx+3, n, :]), dim=0)
        new_data = new_data.unsqueeze(1)  # Add a second dimension to match dumped_values shape
    
        # Concatenate new_data to dumped_values
        dumped_values = torch.cat((dumped_values, new_data), dim=1)  # Concatenate along the second dimension
    #print(dumped_values.shape)
    #Next, generate the "inside box" matrix operator and apply it
    #print("filtering by inside box: 1")
    inside_box_mat = Inside_Box(path_matrix[:i+1,...], dimensions)
    #print(inside_box_mat.shape)
    #print("filtering by inside box: 2")
    first_false = first_false_index(inside_box_mat)#.cpu().numpy()
    #print(len(first_false[first_false==-1]))
    #print("filtering by inside box: 3")
    #outside_box_filter = torch.tensor(generate_path_filter(first_false,M)).cuda()
    outside_box_filter = generate_path_filter(first_false,M)
    
    #print(outside_box_filter.shape)
    #print("filtering by inside box: 4")
    diff_matrix_box = path_matrix * outside_box_filter - path_matrix
    #box_indices = np.where(np.any(diff_matrix_box.cpu().numpy() != 0, axis=(0, 2)))[0]
    box_indices = torch.where(torch.any(torch.any(diff_matrix_box != 0, dim=0), dim=1))[0]

    box_stopping_points = torch.sum(outside_box_filter,dim=(0,2))-1
    for n in box_indices:
        # Extract the first 3 and last 3 rows for column n
        idx = box_stopping_points[n]
        new_data = torch.cat((path_matrix[:3, n, :], path_matrix[idx:idx+3, n, :]), dim=0)
        new_data = new_data.unsqueeze(1)  # Add a second dimension to match dumped_values shape
        # Concatenate new_data to dumped_values
        dumped_values = torch.cat((dumped_values, new_data), dim=1)  # Concatenate along the second dimension
    #filter_indices = np.unique(np.concatenate((path_indices,box_indices)))
    filter_indices = torch.unique(torch.concatenate((path_indices, box_indices)))

    print(f"Amount of streamlines filtered: {len(path_indices)}, {len(box_indices)}")
    #path_mat = np.delete(path_matrix.cpu().numpy(), filter_indices, axis=1)
    #path_filt = np.delete(path_filter.cpu().numpy(), filter_indices, axis=1)
    path_mat = torch.cat([path_matrix[:, :idx], path_matrix[:, idx+1:]], dim=1)
    path_filt = torch.cat([path_filter[:, :idx], path_filter[:, idx+1:]], dim=1)

    #return torch.tensor(path_mat).cuda(), torch.tensor(dumped_values).cuda(), torch.tensor(path_filt).cuda()
    return path_mat, dumped_values, path_filt

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

#@profile
def main():
    options = {
        "path_to_pqr": "/home/santiagovargas/dev/CPET-python/tests/test_files/test_large.pqr",
        "center": [104.785, 113.388, 117.966],
        "x": [105.785, 113.388, 117.966],
        "y": [104.785, 114.388, 117.966],
        "n_samples": 10000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.001,
        "GPU_batch_freq": 100
    }
    x, Q = parse_pqr(options["path_to_pqr"])
    Q=torch.tensor(Q).cuda()
    center = np.array(options["center"])
    x_vec_pt = np.array(options["x"])
    y_vec_pt = np.array(options["y"])
    dimensions = np.array(options["dimensions"])
    step_size = options["step_size"]
    n_samples = options["n_samples"]
    GPU_batch_freq = options["GPU_batch_freq"]
    path_matrix, transformation_matrix, M, path_filter, random_max_samples = initialize_streamline_grid(center, x_vec_pt, y_vec_pt, dimensions, n_samples, step_size)
    path_matrix_torch=torch.tensor(path_matrix).cuda()
    path_filter=torch.tensor(path_filter).cuda()
    dumped_values=torch.tensor(np.empty((6,0,3))).cuda()
    x = (x-center)@np.linalg.inv(transformation_matrix)
    x=torch.tensor(x).cuda()
    j=0
    start_time = time.time()

    for i in range(len(path_matrix)):
      if i % 100 == 0:
        print(i)

      if j == len(path_matrix)-1:
        break
      path_matrix_torch = propagate_topo_matrix(path_matrix_torch,i, x, Q, step_size)
      if i%GPU_batch_freq==0 and i>5:
          path_matrix_torch, dumped_values, path_filter= batched_filter(path_matrix_torch, dumped_values, i,dimensions, M, path_filter, current=True)
          #GPU_batch_freq *= 2
      j+=1
      torch.cuda.empty_cache()
      if dumped_values.shape[1]>=n_samples:
          break
    #path_matrix_torch, dumped_values, path_filter= batched_filter(path_matrix_torch, dumped_values, i,dimensions, M, path_filter, current=False)
    print(dumped_values[:,0,:])
    distances, curvatures = compute_curv_and_dist_mat(dumped_values[0,:,:], dumped_values[1,:,:], dumped_values[2,:,:],dumped_values[3,:,:],dumped_values[4,:,:],dumped_values[5,:,:])
    end_time = time.time()
    print(f"Time taken for {options['n_samples']} calculations with N~{Q.shape}: {end_time - start_time:.2f} seconds")
    topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
    np.savetxt("hist_cpet_mat.txt", topology)
    return topology

main()