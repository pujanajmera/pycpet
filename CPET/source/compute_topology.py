import numpy as np
import time
from numba import jit
from CPET.utils.fastmath import power, nb_subtract

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
    R = nb_subtract(x_0,x)
    R_sq=R**2
    r_mag_sq = np.einsum('ij,ij->i', R_sq, R_sq)[:, np.newaxis]
    r_mag_cube = power(r_mag_sq,3/2)
    E = np.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q)*14.3996451
    return E

#profile
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
    E = calculate_electric_field(x_0, x, Q) #Compute field
    E = E/np.linalg.norm(E)
    x_0 = x_0 +step_size*E
    return x_0

def curv(v_prime, v_prime_prime):
    """
    Computes curvature of the streamline at a given point
    Takes
        v_prime(array) - first derivative of streamline at the point of shape (1,3)
        v_prime_prime(array) - second derivative of streamline at the point of shape (1,3)
    Returns
        curvature(float) - the curvature
    """
    curvature = np.linalg.norm(np.cross(v_prime, v_prime_prime))/np.linalg.norm(v_prime)**3
    return curvature

def compute_curv_and_dist(x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus):
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
    curv_init = curv(x_init_plus-x_init,x_init_plus_plus-2*x_init_plus+x_init)
    curv_final = curv(x_0_plus-x_0,x_0_plus_plus-2*x_0_plus+x_0)
    curv_mean = (curv_init+curv_final)/2
    dist = np.linalg.norm(x_init-x_0)
    return [dist, curv_mean]

def parse_pqr(path_to_pqr):
    """
    Parses pqr file to obtain charges and positions of charges (beta, removes charges that are 0)
    Takes
        path_to_pqr(str) - path to pqr file
    Returns
        np.array(x)(array) - coordinates of charges of shape (N,3)
        np.array(Q).reshape(-1,1) - magnitude and sign of charges of shape (N,1)
    """
    x = []
    Q = []
    with open(path_to_pqr) as pqr_file:
        lines = pqr_file.readlines()
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            coords = [line[31:39].strip(),line[40:48].strip(),line[49:57].strip()]

            charge = line[58:63].strip()
            try:
                tempq = float(charge)
                temp = [float(_) for _ in coords]
            except:
                print(f"Charge or coordinates is not a useable number. Check pqr file formatting for the following line: {line}")
            x.append(temp)
            Q.append(tempq)
    return np.array(x), np.array(Q).reshape(-1,1)

def initialize_box_points(center,
                          x,
                          y,
                          dimensions,
                          n_samples,
                          step_size):
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
    transformation_matrix = np.column_stack([x_unit, y_unit, z_unit]).T  # Each column is a unit vector
    max_distance = 2*np.linalg.norm(np.array(dimensions)) #Define maximum sample limit as 2 times the diagonal
    random_max_samples = np.random.randint(1, max_distance / step_size, n_samples)
    return random_points_local, random_max_samples, transformation_matrix

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
        -half_length <= local_point[0] <= half_length and
        -half_width <= local_point[1] <= half_width and
        -half_height <= local_point[2] <= half_height
    )
    return is_inside

def main():
    options={"path_to_pqr":"./1_wt_run1_0.pqr",
             "center": [55.965,46.219,22.123],
             "x": [56.191,48.344,22.221],
             "y": [57.118,46.793,20.46],
             "n_samples": 100,
             "dimensions": [1.5,1.5,1.5],
             "step_size": 0.001}
    x, Q = parse_pqr(options["path_to_pqr"])
    center = np.array(options["center"])
    x_vec_pt = np.array(options["x"])
    y_vec_pt = np.array(options["y"])
    dimensions = np.array(options["dimensions"])
    step_size = options["step_size"]
    n_samples = options["n_samples"]
    random_start_points, random_max_samples, transformation_matrix = initialize_box_points(center,
                                                                                           x_vec_pt,
                                                                                           y_vec_pt,
                                                                                           dimensions,
                                                                                           n_samples,
                                                                                           step_size)
    hist = []
    start_time = time.time()
    count = 0
    x = (x-center)@np.linalg.inv(transformation_matrix)
    for idx, i in enumerate(random_start_points):
        x_0 = i
        x_init = x_0
        n_iter = random_max_samples[idx]
        for j in range(n_iter):
            x_0 =  propagate_topo(x_0, x, Q, step_size)
            if not Inside_Box(x_0,
                              dimensions):
                count += 1
                break
        x_init_plus = propagate_topo(x_init, x, Q, step_size)
        x_init_plus_plus = propagate_topo(x_init_plus, x, Q, step_size)
        x_0_plus = propagate_topo(x_0, x, Q, step_size)
        x_0_plus_plus = propagate_topo(x_0_plus, x, Q, step_size)
        hist.append(compute_curv_and_dist(x_init, x_init_plus, x_init_plus_plus, x_0, x_0_plus, x_0_plus_plus))
    end_time = time.time()
    np.savetxt("hist_cpet.txt", hist)
    print(f"Time taken for {options['n_samples']} calculations with N~4000: {end_time - start_time:.2f} seconds")
    print(count, len(random_start_points))

main()
