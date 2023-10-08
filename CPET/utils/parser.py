import numpy as np

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
