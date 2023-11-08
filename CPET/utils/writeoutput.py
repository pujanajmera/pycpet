import numpy as np


def write_field_to_file(grid_points, field_points, filename):
    """
    Write grid points and field points to a files
    Takes:
        grid_points(array): 3D grid points (shape: (M,M,M 3))
        field_points(array): Corresponding field values (shape: (M,M,M,3))
        filename(str): Name of the file to write
    """
    # Flatten the grid and field points arrays and stack them along the last axis
    data = np.column_stack((grid_points.reshape(-1, 3), field_points.reshape(-1, 3)))
    # Create a format string for writing each line of the file
    format_str = " ".join(["%f"] * 6)
    # Open the file for writing
    with open(filename, "w") as f:
        # Write 7 lines of hashtags
        for _ in range(7):
            f.write("#\n")
        # Write the data to the file
        np.savetxt(f, data, fmt=format_str)
