import numpy as np


def get_transformation(center, x, y):
    # Convert lists to numpy arrays
    x = x - center  # Translate to origin
    y = y - center  # Translate to origin
    # Normalize the vectors
    x_unit = x / np.linalg.norm(x)
    y_unit = y / np.linalg.norm(y)
    # Calculate the z unit vector by taking the cross product of x and y
    z_unit = np.cross(x_unit, y_unit)
    z_unit = z_unit / np.linalg.norm(z_unit)
    # Recalculate the y unit vector
    y_unit = np.cross(z_unit, x_unit)
    y_unit = y_unit / np.linalg.norm(y_unit)
    transformation_matrix = np.column_stack([x_unit, y_unit, z_unit]).T
    return transformation_matrix
