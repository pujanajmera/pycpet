import numpy as np
import json

def options_parsing(path_to_options):
    with open(path_to_options, 'r') as file:
        options = json.load(file)
    return options

def filter_pqr_radius(x, Q, center, radius=2.0):
    # Filter out point charges that are within a certain radius of the center
    x_recentered = x - center
    r = np.linalg.norm(x_recentered, axis=1)
    
    mask = r > radius
    # remove masked points
    x_filtered = x[mask]
    Q_filtered = Q[mask]

    return x_filtered, Q_filtered

def filter_pqr_radius_resid(x, Q, center, resids, radius=2.0):
    #Filter out entire residues that are outside a certain radius of the center (keeps border res)
    x_recentered = x - center
    r = np.linalg.norm(x_recentered, axis=1)
    resids = np.array(resids)

    #Define list of unique residues to keep
    resid_to_keep_index = []
    for i in range(len(x_recentered)):
        r = np.linalg.norm(x_recentered[i])
        if r < radius:
            resid_to_keep_index.append(i)
    mask = []

    #Create mask with those residues
    for i in range(len(resids)):
        if resids[i] in resid_to_keep_index:
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)
    x_filtered = x_recentered[mask]
    Q_filtered = Q[mask]

    return x_filtered, Q_filtered

def filter_pqr_residue(x, Q, resnames, filter_list):
    # Filter out defined residues
    x = x
    filter_inds = []
    for resname in resnames: 
        if resname in filter_list:
            filter_inds.append(False)
        else: 
            filter_inds.append(True)
    x_filtered = x[filter_inds]
    Q_filtered = Q[filter_inds]

    return x_filtered, Q_filtered


def filter_pqr_atom_num(x, Q, atom_num_list, filter_list):
    # Filter out points that are inside the box
    x_filtered = []
    Q_filtered = []
    for i in range(len(x)):
        if atom_num_list[i] not in filter_list:
            x_filtered.append(x[i])
            Q_filtered.append(Q[i])

    return x_filtered, Q_filtered

def parse_pqr(path_to_pqr, ret_atom_names=False, ret_residue_names=False, ret_resid=False):
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
    resid = []

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
            
            if ret_resid:
                if split_tf:
                    resid.append(line.split()[3])
                else:
                    resid.append(line.split()[4])

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
    
    if ret_resid:
        return np.array(x), np.array(Q).reshape(-1,1), res_id

    return np.array(x), np.array(Q).reshape(-1, 1)
