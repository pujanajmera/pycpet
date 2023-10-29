import numpy as np


def filter_pqr(x, Q, center, radius=2.0):
    # Filter out points that are inside the box
    x = x - center
    r = np.linalg.norm(x, axis=1)
    mask = r > radius
    # remove masked points
    x_filtered = x[mask]
    Q_filtered = Q[mask]

    return x_filtered, Q_filtered


def parse_pqr(path_to_pqr, ret_atom_names=False):
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
    ret_atom_num = []
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
            if tempq != 0:
                x.append(temp)
                Q.append(tempq)
            # clear temp variables
            temp = []
            tempq = []
    if ret_atom_names:
        return np.array(x), np.array(Q).reshape(-1, 1), ret_atom_num
    print(np.array(x)[-20:], np.array(Q).reshape(-1,1)[-20:])
    print(np.sum(np.array(Q)))
    return np.array(x), np.array(Q).reshape(-1, 1)
