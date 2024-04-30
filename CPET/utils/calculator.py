import numpy as np
import torch
import pkg_resources
import matplotlib
import matplotlib.pyplot as plt

package_name = "CPET-python"
package = pkg_resources.get_distribution(package_name)
package_path = package.location
# import cupy as cp

from CPET.utils.fastmath import nb_subtract, power, nb_norm, nb_cross
from CPET.utils.c_ops import Math_ops
Math = Math_ops(shared_loc=package_path + "/CPET/utils/math_module.so")


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
    epsilon = 10e-7
    E = calculate_electric_field(x_0, x, Q)
    #if np.linalg.norm(E) > epsilon: 
    E = E / (np.linalg.norm(E) + epsilon)
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
    epsilon = 10e-7
    E = calculate_electric_field_dev_c_shared(x_0, x, Q, debug)
    #if np.linalg.norm(E) > epsilon: 
    E = E / (np.linalg.norm(E) + epsilon)
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
    epsilon = 10e-6

    if mask_list is None:
        E_full = calculate_electric_field_dev_c_shared_batch(x_0_list, x, Q)
        #print("passed first iter")
    else: 
        # if theres a mask, only calculate the field for the points that are inside the box
        # filter x_0_list on the mask
        
        x_0_list_filtered = [x_0 for x_0, mask in zip(x_0_list, mask_list) if not mask]
        E = calculate_electric_field_dev_c_shared_batch(x_0_list_filtered, x, Q)
        # expand E to the original size E is 0 for masked points
        #print("mask list: {}".format(mask_list))
        #print("E: {}".format(E))
        E_full = []
        #print("E: {}".format(E))
        #print("type {}".format(type(E[0])))
        ind_filtered = 0 
        for ind, x in enumerate(x_0_list): 
            if mask_list[ind] == False: 
                E_full.append(E[ind_filtered])
                ind_filtered += 1
            else: 
                E_full.append([0.0, 0.0, 0.0])
                
    assert len(E_full) == len(x_0_list), "efull len is funky"
    
    #print("efull: {}".format(E_full))
    E_list = [e / (np.linalg.norm(e) + epsilon) for e in E_full]
    x_0_list = [x_0 + step_size * e for x_0, e in zip(x_0_list, E_list)]
    #print(x_0_list)
    return x_0_list


def initialize_box_points_random(center, x, y, dimensions, n_samples, step_size):
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


def initialize_box_points_uniform(center, x, y, step_size, dimensions):
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

    transformation_matrix = np.column_stack(
        [x_unit, y_unit, z_unit]
    ).T  # Each column is a unit vector

    # construct a grid of points in the box - lengths are floats
    x_coords = np.arange(-half_length, half_length, step_size)
    y_coords = np.arange(-half_width, half_width, step_size)
    z_coords = np.arange(-half_height, half_height, step_size)

    x_mesh, y_mesh, z_mesh = np.meshgrid(x_coords, y_coords, z_coords)
    local_coords = np.stack([x_mesh, y_mesh, z_mesh], axis=-1)
    
    return local_coords, transformation_matrix


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


def calculate_electric_field_dev_c_shared(x_0, x, Q, debug=False):
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
    #print("subtract")

    R = nb_subtract(x_0, x)

    R_sq = R**2
    # r_mag_sq = np.einsum("ij->i", R_sq).reshape(-1, 1)
    #print("rsq shape: {}".format(R_sq.shape))
    # print(R_sq.dtype)
    #print("einsum 1")
    r_mag_sq = Math.einsum_ij_i(R_sq).reshape(-1, 1)
    #print("rmag sq size: {}".format(r_mag_sq.shape))
    #print("power op")
    r_mag_cube = np.power(r_mag_sq, 3 / 2)
    #print("einsum 2")
    E = Math.einsum_operation(R, 1 / r_mag_cube, Q) * 14.3996451
    #print(E.shape)
    # print("-")
    #print("end efield calc")
    return E


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
    R_sq_list = np.array([R**2 for R in R_list])
    r_mag_sq_list = Math.einsum_ij_i_batch(R_sq_list)#.reshape(-1, 1)
    #print("rmag: {}".format(r_mag_sq_list))
    #print("rmag len: {}".format(len(r_mag_sq_list)))
    #print("rmag 1 size: {}".format(r_mag_sq_list[0].shape))
    r_mag_cube_list = np.power(r_mag_sq_list, 3 / 2)
    recip_r_mag_list = [1/val for val in r_mag_cube_list]
    #print("recip r {}".format(recip_r_mag_list))
    E_list = Math.einsum_operation_batch(R_list, recip_r_mag_list, Q, batch_size) * 14.3996451
    #print("passed einsum op")
    # print(E.shape)
    #print("-")
    #print("Elist: {}".format(E_list))
    return E_list


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


"""
def calculate_electric_field_cupy(x_0, x, Q):
    #Computes electric field at a point given positions of charges
    #Takes
    #    x_0(array) - position to compute field at of shape (N,3)
    #    x(array) - positions of charges of shape (N,3)
    #    Q(array) - magnitude and sign of charges of shape (N,1)
    #Returns
    #    E(array) - electric field at the point of shape (1,3)
    
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
"""


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
        r_mag_cube = 1 / np.power(r_mag_sq, 3 / 2)
        # compute field using einsum 
        E[i] = np.einsum("ij,ij,ij->j", R, r_mag_cube, Q) * 14.3996451
        # compute without einsum
        #E[i] = np.sum(R * r_mag_cube * Q, axis=0) * 14.3996451
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
    #print("inside box")
    # Convert lists to numpy arrays
    half_length, half_width, half_height = dimensions
    # Check if the point lies within the dimensions of the box
    is_inside = (
        -half_length <= local_point[0] <= half_length
        and -half_width <= local_point[1] <= half_width
        and -half_height <= local_point[2] <= half_height
    )
    return is_inside


def distance_numpy(hist1, hist2):
    a = (hist1 - hist2) ** 2
    b = hist1 + hist2
    return np.sum(np.divide(a, b, out=np.zeros_like(a), where=b != 0)) / 2.0


def mean_and_curve_to_hist(mean_dist, curve): 
    #Calculate reasonable maximum distances and curvatures
    #curvatures, distances = [],[]
    max_distance = max(mean_dist)
    max_curvature = max(curve)
    
    # bins is number of histograms bins in x and y direction (so below is 200x200 bins)
    # range gives xrange, yrange for the histogram
    a, b, c, q = plt.hist2d(
        mean_dist,
        curve,
        bins=200,
        range=[[0, max_distance], [0, max_curvature]],
        norm=matplotlib.colors.LogNorm(),
        density=True,
        cmap="jet",
    )

    NormConstant = 0
    for j in a:
        for m in j:
            NormConstant += m

    actual = []
    for j in a:
        actual.append([m / NormConstant for m in j])

    actual = np.array(actual)
    histogram = actual.flatten()
    return np.array(histogram)


def make_histograms(topo_files, plot=False):
    histograms = []

    #Calculate reasonable maximum distances and curvatures
    dist_list = []
    curv_list = []
    for topo_file in topo_files:
        curvatures, distances = [],[]
        print(topo_file)
        with open(topo_file) as topology_data:
            for line in topology_data:
                if line.startswith("#"):
                    continue

                line = line.split()
                distances.append(float(line[0]))
                curvatures.append(float(line[1]))
        print(max(distances),max(curvatures))
        dist_list.extend(distances)
        curv_list.extend(curvatures)
    print(len(dist_list))
    print(len(curv_list))

    #Do 95th percentiles instead to take care of extreme cases for curvature
    max_distance = max(dist_list)
    #max_distance = np.percentile(dist_list, 95)
    max_curvature = max(curv_list)
    #max_curvature = np.percentile(curv_list, 98)
    print(f"Max distance: {max_distance}")
    print(f"Max curvature: {max_curvature}")
    #Make histograms
    for topo_file in topo_files:
        curvatures, distances = [], []

        with open(topo_file) as topology_data:
            for line in topology_data:
                if line.startswith("#"):
                    continue

                line = line.split()
                distances.append(float(line[0]))
                curvatures.append(float(line[1]))

        # bins is number of histograms bins in x and y direction (so below is 100x100 bins)
        # range gives xrange, yrange for the histogram
        a, b, c, q = plt.hist2d(
            distances,
            curvatures,
            bins=200,
            range=[[0, max_distance], [0, max_curvature]],
            norm=matplotlib.colors.LogNorm(),
            density=True,
            cmap="jet",
        )

        NormConstant = 0
        for j in a:
            for m in j:
                NormConstant += m

        actual = []
        for j in a:
            actual.append([m / NormConstant for m in j])

        actual = np.array(actual)
        histograms.append(actual.flatten())
        if plot:
            plt.show()

    return np.array(histograms)


def construct_distance_matrix(histograms):
    matrix = np.diag(np.zeros(len(histograms)))
    for i, hist1 in enumerate(histograms):
        for j, hist2 in enumerate(histograms[i + 1 :]):
            j += i + 1
            matrix[i][j] = distance_numpy(hist1, hist2)
            matrix[j][i] = matrix[i][j]
    return matrix



