def initialize_grid(center, x, y, density, dimensions, offset):
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
    # Create arrays of coordinates along each axis
    if offset == "center":
        x_coords = np.linspace(-dimensions[0], dimensions[0], 2*density[0]+1)
        y_coords = np.linspace(-dimensions[1], dimensions[1], 2*density[1]+1)
        z_coords = np.linspace(-dimensions[2], dimensions[2], 2*density[2]+1)
    else:
        x_coords = np.linspace(0, dimensions[0], density[0]+1)
        y_coords = np.linspace(0, dimensions[1], density[1]+1)
        z_coords = np.linspace(0, dimensions[2], density[2]+1)
    # Create a meshgrid of coordinates
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    # Stack the meshgrid coordinates into a single array of shape (density[0], density[1], density[2], 3)
    local_coords = np.stack([x_mesh, y_mesh, z_mesh], axis=-1)
    return local_coords, transformation_matrix


center = np.array([55.965,46.219,22.123])
x = np.array([56.191,48.344,22.221])
y = np.array([57.118,46.793,20.46])
dimensions = np.array([1.5,1.5,1.5])
density = np.array([10,10,10])

grid_points, transformation_matrix = initialize_grid(center, x, y, density, dimensions, "center")
x, Q = parse_pqr("1_wt_run1_0.pqr")
x = (x-center)@np.linalg.inv(transformation_matrix)
field_points = compute_field_on_grid(grid_points, x, Q)
write_field_to_file(grid_points, field_points, "1_wt_run1_0_efield.dat")
