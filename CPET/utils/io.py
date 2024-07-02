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


def save_numpy_as_dat(meta_data, field, name):
    """
    Saves np array in original format from cpet output
    Takes:
        meta_data: dictionary with meta data
        field: np array with average field
        name: name of file to save
    """

    dimensions = meta_data["dimensions"]
    step_size_list = meta_data["num_steps"]
    
    first_line = "#Sample Density: {} {} {} Volumne: Box: {} {} {} \n".format(
        int((step_size_list[0] - 1 ) / 2), 
        int((step_size_list[1] - 1 ) / 2),
        int((step_size_list[2] - 1 ) / 2),
        dimensions[0],
        dimensions[1],
        dimensions[2]
    )
    
    lines_header = [first_line]
    # add six lines starting with #
    lines_header = lines_header + ["#\n"] * 6

    # write as
    np.savetxt(
        name,
        field,
        fmt="%.3f",
    )

    # open file and write header
    with open(name, "r") as f:
        lines = f.readlines()

    with open(name, "w") as f:
        f.write("".join(lines_header))
        f.write("".join(lines))


def read_mat(file, meta_data=False, verbose=False):
    """
    Pulls the matrix from a cpet file
    Takes
        file: cpet file
        meta_data(Optionally): returns the meta data
    Returns
        mat: matrix of xyz coordinates
        meta_data(Optionally): dictionary of meta data
    """
    with open(file) as f:
        lines = f.readlines()

    steps_x = 2 * int(lines[0].split()[2]) + 1
    steps_y = 2 * int(lines[0].split()[3]) + 1
    steps_z = 2 * int(lines[0].split()[4]) + 1
    x_size = float(lines[0].split()[-3])
    y_size = float(lines[0].split()[-2])
    z_size = float(lines[0].split()[-1])
    step_size_x = np.round(x_size / float(lines[0].split()[2]), 4)
    step_size_y = np.round(y_size / float(lines[0].split()[3]), 4)
    step_size_z = np.round(z_size / float(lines[0].split()[4]), 4)

    meta_dict = {
        "first_line": lines[0],
        "steps_x": steps_x,
        "steps_y": steps_y,
        "steps_z": steps_z,
        "step_size_x": step_size_x,
        "step_size_y": step_size_y,
        "step_size_z": step_size_z,
        "bounds_x": [-x_size, x_size + step_size_x],
        "bounds_y": [-y_size, y_size + step_size_y],
        "bounds_z": [-z_size, z_size + step_size_z],
    }
    #print(lines[0].split())
    #print(meta_dict)
    if verbose:
        print(meta_dict)

    if meta_data:
        return meta_dict

    else:
        steps_x = 2 * int(lines[0].split()[2]) + 1
        steps_y = 2 * int(lines[0].split()[3]) + 1
        steps_z = 2 * int(lines[0].split()[4]) + 1
        mat = np.zeros((steps_x, steps_y, steps_z, 3))
        #print(mat.shape)
        
        for ind, i in enumerate(lines[7:]):
            line_split = i.split()
            # print(i)
            mat[
                int(ind / (steps_z * steps_y)),
                int(ind / steps_z % steps_y),
                ind % steps_z,
                0,
            ] = float(line_split[-3])
            mat[
                int(ind / (steps_z * steps_y)),
                int(ind / steps_z % steps_y),
                ind % steps_z,
                1,
            ] = float(line_split[-2])
            mat[
                int(ind / (steps_z * steps_y)),
                int(ind / steps_z % steps_y),
                ind % steps_z,
                2,
            ] = float(line_split[-1])

        return mat


def default_options_initializer(options): 
    """
        Initializes default options for CPET after checking if they are present in the options dictionary
        Takes
            options(dict) - dictionary of options
        Returns
            options(dict) - dictionary of options with default values
    """
    
    if "concur_slip" not in options.keys():
        options["concur_slip"] = 4

    if "dtype" not in options.keys():
        options["dtype"] = "float32"

    if "GPU_batch_freq" not in options.keys():
        options["GPU_batch_freq"] = 100

    if "step_size" not in options.keys():
        options["step_size"] = 0.1

    if "n_samples" not in options.keys():
        options["n_samples"] = 10000

    if "profile" not in options.keys():
        options["profile"] = False

    return options


def initialize_box_points(center, x, y, dimensions, n_samples, step_size):
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


def filter_radius(x, Q, center, radius=2.0):
    # Filter out points that are inside the box
    x_recentered = x - center
    r = np.linalg.norm(x_recentered, axis=1)
    # print("radius filtering {}".format(radius))
    mask = r < radius
    # remove masked points
    x_filtered = x[mask]
    Q_filtered = Q[mask]
    print("radius filter leaves: {}".format(len(Q_filtered)))
    # print(np.linalg.norm(x_filtered, axis=1))
    return x_filtered, Q_filtered


def filter_residue(x, Q, resids, filter_list):
    # Filter out points that are inside the box
    x = x
    filter_inds = []
    for resid in resids:
        if resid in filter_list:
            filter_inds.append(False)
        else:
            filter_inds.append(True)
    x_filtered = x[filter_inds]
    Q_filtered = Q[filter_inds]

    return x_filtered, Q_filtered


def filter_resnum(x, Q, resnums, filter_list):
    # Filter out points that are inside the box
    x = x
    filter_inds = []
    for resnum in resnums:
        if resnum in filter_list:
            filter_inds.append(False)
        else:
            filter_inds.append(True)
    x_filtered = x[filter_inds]
    Q_filtered = Q[filter_inds]

    return x_filtered, Q_filtered


def filter_resnum_andname(x, Q, resnums, resnames, filter_list):
    # Filter out points that are inside the box
    x = x
    filter_inds = []
    for i in range(len(resnums)):
        if {str(resnums[i]): resnames[i]} in filter_list:
            filter_inds.append(False)
        else:
            filter_inds.append(True)
    x_filtered = x[filter_inds]
    Q_filtered = Q[filter_inds]

    return x_filtered, Q_filtered


def filter_in_box(x, Q, center, dimensions):
    x_recentered = x - center
    print("Filtering Charges in Sampling Box")
    # Filter out points that are inside the box
    limits = {
        "x": [-dimensions[0], dimensions[0]],
        "y": [-dimensions[1], dimensions[1]],
        "z": [-dimensions[2], dimensions[2]],
    }
    # print("box dimensions: {}".format(limits))
    # print(x.shape)
    mask_x = (x_recentered[:, 0] > limits["x"][0]) & (
        x_recentered[:, 0] < limits["x"][1]
    )
    mask_y = (x_recentered[:, 1] > limits["y"][0]) & (
        x_recentered[:, 1] < limits["y"][1]
    )
    mask_z = (x_recentered[:, 2] > limits["z"][0]) & (
        x_recentered[:, 2] < limits["z"][1]
    )

    mask = mask_x & mask_y & mask_z

    # only keep points that are outside the box
    x_filtered = x[~mask]
    Q_filtered = Q[~mask]
    # print("masked points: {}".format(len(mask)))
    return x_filtered, Q_filtered


def filter_atom_num(x, Q, atom_num_list, filter_list):
    # Filter out points that are inside the box
    x_filtered = []
    Q_filtered = []
    for i in range(len(x)):
        if atom_num_list[i] not in filter_list:
            x_filtered.append(x[i])
            Q_filtered.append(Q[i])

    return x_filtered, Q_filtered


def parse_pqr(path_to_pqr, ret_atom_names=False, ret_residue_names=False):
    """
    Parses pqr file to obtain charges and positions of charges (beta, removes charges that are 0)
    Takes
        path_to_pqr(str) - path to pqr file
        ret_atom_names(bool) - whether to return atom names
        ret_residue_names(bool) - whether to return residue names
    Returns
        np.array(x)(array) - coordinates of charges of shape (N,3)
        np.array(Q).reshape(-1,1) - magnitude and sign of charges of shape (N,1)
    """
    x = []
    Q = []
    ret_atom_num = []
    res_name = []

    with open(path_to_pqr) as pqr_file:
        lines = pqr_file.readlines()

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            shift = 0
            res_ind = 5  # index of residue value
            split_tf = False
            if len(line.split()[0]) > 6:
                res_ind = 4
                split_tf = True

            if ret_atom_names:
                if split_tf:
                    # remove HETATM from split 0
                    ret_atom_num.append(int(line.split()[0][6:]))
                else:
                    ret_atom_num.append(int(line.split()[1]))

            if ret_residue_names:
                if split_tf:
                    res_name.append(line.split()[2])
                else:
                    res_name.append(line.split()[3])

            if len(line.split()[res_ind]) > 4:
                res_val = int(line.split()[res_ind - 1][1:])
            else:
                res_val = int(line.split()[res_ind])

            if res_val > 999:
                shift += int(np.log10(res_val) - 2)

            coords = [
                line[30 + shift : 38 + shift],
                line[38 + shift : 46 + shift],
                line[46 + shift : 54 + shift],
            ]

            if shift == -1:
                print("----------------------")
                print("coords: ", coords)

            coords = [_.strip() for _ in coords]
            charge = line[54 + shift : 61 + shift]

            if shift == -1:
                print("charge: ", [charge])

            charge = charge.strip()

            try:
                tempq = float(charge)
                temp = [float(_) for _ in coords]

            except:
                print(
                    f"Charge or coordinates is not a useable number. Check pqr file formatting for the following line: {line}"
                )

            assert temp != [], "charge incorrectly parsed"
            assert tempq != [], "coord incorrectly parsed"
            x.append(temp)
            Q.append(tempq)
            # clear temp variables
            temp = []
            tempq = []
    if ret_atom_names:
        return np.array(x), np.array(Q).reshape(-1, 1), ret_atom_num

    if ret_residue_names:
        return np.array(x), np.array(Q).reshape(-1, 1), res_name

    return np.array(x), np.array(Q).reshape(-1, 1)


def parse_pdb(pdb_file_path, get_charges=False, float32=True):
    """
    Takes:
        pdb_file_path(str) - path to pdb file
        get_charges(bool) - whether to parse/return charges

    Returns:
        atom_info(list of tuples) - containing information about each atom in the pdb file
    """

    xyz = []
    Q = []
    atom_number = []
    residue_name = []
    residue_number = []
    atom_type = []
    if float32:
        dtype = "float32"
    else:
        dtype = "float64"

    with open(pdb_file_path, "r") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_number_line = int(line[6:11].strip())
                atom_type_line = line[12:16].strip()
                residue_name_line = line[17:20].strip()
                residue_number_line = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                # atom_info.append((atom_number, atom_type, residue_name, residue_number, x, y, z))

                xyz.append([x, y, z])
                atom_number.append(atom_number_line)
                residue_name.append(residue_name_line)
                residue_number.append(residue_number_line)
                atom_type.append(atom_type_line)

                if get_charges:
                    Q.append(float(line[55:64].strip()))

    if get_charges:
        return (
            np.array(xyz, dtype=dtype),
            np.array(Q, dtype=dtype).reshape(-1, 1),
            np.array(atom_number, dtype=dtype),
            np.array(residue_name),
            np.array(residue_number),
            np.array(atom_type),
        )
    return (
        np.array(xyz, dtype=dtype),
        np.array(atom_number, dtype=dtype),
        np.array(residue_name),
        np.array(residue_number),
        np.array(atom_type),
    )


def calculate_center(coordinates, method):
    """
    Helper to calculate the center of a list of atoms
    Takes:
        atoms(list) - list of atoms
        method(str) - method to calculate the center
    Returns:
        np.array(center) - center of the atoms
    """

    if method == "mean":
        return np.mean(coordinates, axis=0)
    elif method == "first":
        return coordinates[0]
    elif method == "inverse":
        first_atom = coordinates[0]
        average_of_others = np.mean(coordinates[1:], axis=0)
        return 2 * average_of_others - first_atom
