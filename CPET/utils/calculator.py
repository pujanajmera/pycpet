import numpy as np
from CPET.utils.fastmath import nb_subtract

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
    r_mag_cube = np.power(r_mag_sq,3/2)
    E = np.einsum("ij,ij,ij->j", R, 1 / r_mag_cube, Q)*14.3996451
    return E

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
