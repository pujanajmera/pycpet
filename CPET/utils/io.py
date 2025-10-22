import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import logging

log = logging.getLogger(__name__)


def get_input_files(inputpath, filetype):
    """
    Check if the pdb/pqr folder contains any pdb/pqr files
    Parameters:
        pdb_folder (str): Path to the folder containing pdb/pqr
        filetype (str): Type of input files ('pdb' or 'pqr')
    Returns:
        files_input (list): List of pdb/pqr files in the folder
    """
    if filetype not in ["pdb", "pqr"]:
        raise ValueError("filetype must be 'pdb' or 'pqr'")
    files_input = glob(inputpath + f"/*.{filetype}")
    if len(files_input) == 0:
        raise ValueError("No pdb files found in the input directory")
    if len(files_input) == 1:
        log.warning("Only one pdb file found in the input directory")
    return files_input


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


def save_numpy_as_dat(meta_data, volume, name):
    """
    Saves np array in original format from cpet output
    Takes:
        meta_data: dictionary with meta data
        field: np array with average field
        name: name of file to save
    """

    dimensions = meta_data["dimensions"]
    num_steps_list = meta_data["num_steps"]
    trans_mat = meta_data["transformation_matrix"].transpose()
    center = meta_data["center"]

    first_line = "#Sample Density: {} {} {}; Volume: Box: {} {} {}\n".format(
        num_steps_list[0],
        num_steps_list[1],
        num_steps_list[2],
        dimensions[0],
        dimensions[1],
        dimensions[2],
    )
    second_line = "#Frame 0\n"
    third_line = "#Center: {} {} {}\n".format(center[0], center[1], center[2])
    basis_matrix_lines = "#Basis Matrix:\n# {} {} {}\n# {} {} {}\n# {} {} {}\n".format(
        trans_mat[0][0],
        trans_mat[0][1],
        trans_mat[0][2],
        trans_mat[1][0],
        trans_mat[1][1],
        trans_mat[1][2],
        trans_mat[2][0],
        trans_mat[2][1],
        trans_mat[2][2],
    )

    lines_header = [first_line] + [second_line] + [third_line] + [basis_matrix_lines]
    """
    #OPTIONAL FOR TESTING NO TRANSFORMATION
    field_coords = field[:, 0:3]
    field_vecs = field[:, 3:6]
    transformed_field_coords = field_coords@trans_mat.T
    transformed_field_coords = transformed_field_coords + center
    transformed_field_vecs = np.matmul(field_vecs, trans_mat.T)
    field = np.concatenate((transformed_field_coords, transformed_field_vecs), axis=1)
    """

    np.savetxt(
        name,
        volume,
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
    print(lines[0].split())
    print(meta_dict)
    if verbose:
        print(meta_dict)

    if meta_data:
        return meta_dict

    else:
        steps_x = 2 * int(lines[0].split()[2]) + 1
        steps_y = 2 * int(lines[0].split()[3]) + 1
        steps_z = 2 * int(lines[0].split()[4]) + 1
        mat = np.zeros((steps_x, steps_y, steps_z, 3))
        # print(mat.shape)

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


def pull_mats_from_MD_folder(root_dir):
    x = []
    target_files = []
    target_files = glob(root_dir + "/*.dat")

    for i in tqdm(target_files):
        x.append(read_mat(i))

    x = np.array(x)

    return x


def default_options_initializer(options):
    """
    Initializes default options for CPET after checking if they are present in the options dictionary
    Takes
        options(dict) - dictionary of options
    Returns
        options(dict) - dictionary of options with default values
    """
    # Directories:
    if "inputpath" not in options.keys():
        options["inputpath"] = "./inpdir"
    if "outputpath" not in options.keys():
        options["outputpath"] = "./outdir"
    if "inputfiletype" not in options.keys():
        options["inputfiletype"] = "pdb"  # Default to pdb with charges

    # Developer Options:
    if "profile" not in options.keys():
        options["profile"] = False

    # Debugging Options:
    if "write_transformed_pdb" not in options.keys():
        options["write_transformed_pdb"] = False

    # Box options (all 3D calcs):
    if "dimensions" not in options.keys():
        options["dimensions"] = None
    if "step_size" not in options.keys():  # Both for topology/box
        options["step_size"] = None
    if "initializer" not in options.keys():
        options["initializer"] = "uniform"
    if "box_shift" not in options.keys():
        options["box_shift"] = [0, 0, 0]

    # Filtering Options:
    if "filter_intersect" not in options.keys():
        options["filter_intersect"] = False

    # Topology Options:
    if "concur_slip" not in options.keys():
        options["concur_slip"] = 4
    if "dtype" not in options.keys():
        options["dtype"] = "float32"
    if "GPU_batch_freq" not in options.keys():
        options["GPU_batch_freq"] = 100
    if "n_samples" not in options.keys():
        options["n_samples"] = None

    return options


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


def filter_radius_whole_residue(x, Q, resids, resnums, center, radius=2.0):
    """
    Filters out entire residues that have any points that fall outside of the radius
    Takes
        x(array) - coordinates of charges of shape (N,3)
        Q(array) - magnitude and sign of charges of shape (N,1)
        resids(array) - residue ids/names of shape (N,)
        resnums(array) - residue numbers of shape (N,)
        center(array) - center of box of shape (1,3)
        radius(float) - radius to filter
    """
    x_recentered = x - center
    is_in_radius = np.linalg.norm(x_recentered, axis=1) < radius
    resid_current = None
    resnum_current = None
    true_res_dict = {}
    print(x_recentered.shape)
    print(Q.shape)
    print(resids.shape)
    print(resnums.shape)
    j = 0
    # Generate dictionary of indices. Accounts for residues that have the same resnum but different resid, as those aren't adjacent
    for i in range(len(resids)):
        if resids[i] == resid_current and resnums[i] == resnum_current:
            current_entry = true_res_dict[j - 1]
            current_entry["indices"].append(i)
            if is_in_radius[i]:
                current_entry["is_in_radius"] = True
        else:
            true_res_dict[j] = {
                "resid": resids[i],
                "resnum": resnums[i],
                "indices": [i],
                "is_in_radius": is_in_radius[i],
            }
            resid_current = resids[i]
            resnum_current = resnums[i]
            j += 1
    # Compress true_res_dict into a list of indices to filter out
    indices_to_filter_out = [
        k
        for key in true_res_dict
        if not true_res_dict[key]["is_in_radius"]
        for k in true_res_dict[key]["indices"]
    ]
    x_filtered = np.delete(x, indices_to_filter_out, axis=0)
    Q_filtered = np.delete(Q, indices_to_filter_out, axis=0)
    print("radius filter leaves: {}".format(len(Q_filtered)))
    # print(np.linalg.norm(x_filtered, axis=1))
    return x_filtered, Q_filtered


def filter_atomspec(x, Q, ID, filter_dict, intersect):
    """
    General filter to remove any specification based on input
    filter list, either by the overlap of the lists or the union
    of the lists

    Parameters
    ----------
    x : array
        Coordinates of charges of shape (N,3)
    Q : array
        Magnitude and sign of charges of shape (N,1)
    ID : list
        Identity information of shape (N,), where each value is a tuple of
        the form (atom_number, atom_type, resid, resnum, chain). Chain is optional.
    filter_dict : dict
        Dictionary of identity information to filter out
    intersect : bool
        If True, filter out only if all conditions in filter_dict are met (intersection)

    Returns
    -------
    x_filtered : array
        Filtered coordinates of charges of shape (N,3)
    Q_filtered : array
        Filtered magnitude and sign of charges of shape (N,1)
    ID_filtered : list
        Filtered identity information of shape (N,)
    """
    filter_idx = []
    atom_attributes_list = ["atom_number", "atom_type", "resid", "resnum", "chain"]
    atom_attributes_considered = []
    for key in filter_dict.keys():
        if key not in atom_attributes_list:
            raise ValueError(
                f"Key {key} not recognized. Must be one of {atom_attributes_list}"
            )
        atom_attributes_considered.append(key)
    log.info(f"Filtering by attributes: {atom_attributes_considered}")
    if not intersect:  # Default behavior, union of filters
        log.info("Filtering by union of filters (default)")
        for i in atom_attributes_considered:
            filter_vals = filter_dict[i]
            for j in range(len(ID)):
                if ID[j][atom_attributes_list.index(i)] in filter_vals:
                    filter_idx.append(j)
        filter_idx = list(set(filter_idx))  # Remove duplicates
    else:
        log.info("Filtering by an ordered intersection of filters")
        attr_len = []
        for i in atom_attributes_considered:
            attr_len.append(len(filter_dict[i]))
        if len(set(attr_len)) != 1:
            raise ValueError(
                "All filter lists must be the same length when using intersection filtering"
            )
        for i in range(attr_len[0]):
            filter_vals = {
                key: filter_dict[key][i] for key in atom_attributes_considered
            }
            for j in range(len(ID)):
                match = True
                for key in filter_vals:
                    val = filter_vals[key]
                    id_val = ID[j][atom_attributes_list.index(key)]
                    if val != "*" and val != "" and id_val != val:
                        match = False
                        break
                if match:
                    filter_idx.append(j)
        filter_idx = list(set(filter_idx))  # Remove duplicates
    log.debug(f"Length before filtering: {len(x)}, {len(Q)}, {len(ID)}")
    x_filtered = np.delete(x, filter_idx, axis=0)
    Q_filtered = np.delete(Q, filter_idx, axis=0)
    ID_filtered = np.delete(ID, filter_idx, axis=0)
    if len(x_filtered) != len(Q_filtered) or len(x_filtered) != len(ID_filtered):
        raise ValueError("Unexpected: filtered arrays have different lengths")
    log.info(f"Number of atoms filtered out: {len(filter_idx)}")
    return x_filtered, Q_filtered, ID_filtered


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


def parse_pqr(path_to_pqr):
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
    res_num = []
    atom_type = []

    with open(path_to_pqr) as pqr_file:
        lines = pqr_file.readlines()

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            shift = 0
            res_ind = 5  # index of residue number
            split_tf = False
            if len(line.split()[0]) > 6:
                res_ind = 4
                split_tf = True

            if split_tf:
                # remove HETATM from split 0
                ret_atom_num.append(int(line.split()[0][6:]))
                res_name.append(line.split()[2])
                atom_type.append(line.split()[1])

            else:
                ret_atom_num.append(int(line.split()[1]))
                res_name.append(line.split()[3])
                atom_type.append(line.split()[2])

            if len(line.split()[res_ind]) > 4:
                res_val = int(line.split()[res_ind - 1][1:])
            else:
                res_val = int(line.split()[res_ind])

            res_num.append(res_val)

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

    return (
        np.array(x),
        np.array(Q).reshape(-1, 1),
        np.array(ret_atom_num),
        np.array(res_name),
        np.array(res_num),
        np.array(atom_type),
    )


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
    chains = []
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
                chain_line = line[21]
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
                chains.append(chain_line)

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
            np.array(chains),
        )
    return (
        np.array(xyz, dtype=dtype),
        np.array(atom_number, dtype=dtype),
        np.array(residue_name),
        np.array(residue_number),
        np.array(atom_type),
        np.array(chains),
    )


def get_atoms_for_axes(x, atom_type, residue_number, chains, options, seltype="center"):
    if "chains" in options[seltype].keys():
        if type(options[seltype]["chains"]) != list:
            raise ValueError("chains must be a list")
        if len(options[seltype]["chains"]) != len(options[seltype]["atoms"]):
            raise ValueError("chains must be the same length as atoms")
        centering_atoms = [
            (k, v, chain)
            for atom_dict, chain in zip(
                options[seltype]["atoms"], options[seltype]["chains"]
            )
            for k, v in atom_dict.items()
        ]
        log.debug(f"len(chains): {len(chains)}")
        log.debug(f"len(atom_type): {len(atom_type)}")
        log.debug(f"len(residue_number): {len(residue_number)}")
        log.debug(f"len(x): {len(x)}")
        pos_considered = [
            pos
            for atom_res in centering_atoms
            for idx, pos in enumerate(x)
            if (
                atom_type[idx],
                residue_number[idx],
                chains[idx],
            )
            == atom_res
        ]
    else:
        log.debug(options[seltype]["atoms"])
        centering_atoms = [
            (k, v)
            for atom_dict in options[seltype]["atoms"]
            for k, v in atom_dict.items()
        ]
        log.info(f"centering atoms for {seltype}: {centering_atoms}")
        atom_set = set(centering_atoms)
        pos_considered = [
            pos
            for (atype, rnum), pos in zip(zip(atom_type, residue_number), x)
            if (atype, rnum) in atom_set
        ]
    log.debug(f"pos considered for {seltype}: {pos_considered}")
    return pos_considered


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
