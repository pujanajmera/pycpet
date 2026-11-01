from CPET.source.calculator import calculator
from CPET.utils.calculator import distance_numpy

import argparse
import os
import json
import numpy as np


def load_hist(pdb, tops_dir):
    # Load histogram from correpsonding topology file
    pdb_name = pdb[:-4]
    top_name = pdb_name + ".top"
    top_path = os.path.join(tops_dir, top_name)
    with open(top_path, "r") as f:
        lines = f.readlines()
    hist = []
    for line in lines:
        distcurv = line.split()
        hist.append([float(distcurv[0]), float(distcurv[1])])
    hist = np.array(hist)
    return hist


def get_hist_grid(hist, distbound, distance_nbins, curvbound, curvature_nbins):
    a, _, _ = np.histogram2d(
        hist[:, 0],
        hist[:, 1],
        bins=[distance_nbins, curvature_nbins],
        range=[distbound, curvbound],
    )
    NormConstant = np.sum(a)
    hist_grid = a / NormConstant
    return hist_grid.flatten()


def get_residues(resnames, resids):
    """
    List the residues of a structure in order, skipping water (WAT).

    Parameters
    ----------
    resnames, resids : sequence
        Residue name and residue number of every atom, in order.

    Returns
    -------
    list of dict
        One ``{"resname": ..., "resid": ...}`` per residue.

    Raises
    ------
    ValueError
        If a residue shows up again after a different residue.
    """
    residues = []
    residue_prev = {}
    for resname, resid in zip(resnames, resids):
        residue = {"resname": resname, "resid": resid}

        # Atoms of the same residue are next to each other, so only the first one counts.
        # A residue that was already listed but isn't the previous one is an error.
        if residue == residue_prev or resname == "WAT":
            continue
        elif residue in residues:
            raise ValueError("Error: Residue " + str(residue) + " is repeated in the list")
        else:
            residues.append(residue)
            residue_prev = residue
    return residues


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Residue breakdown analysis")
    parser.add_argument("-t", "--tops", help="Topology file directory", required=True)
    parser.add_argument("-p", "--pdbs", help="PDB file directory", required=True)
    parser.add_argument(
        "-c", "--curvbound", help="Histogram curvature bounds", required=True
    )
    parser.add_argument(
        "-cx", "--curvres", help="Histogram curvature resolution", required=True
    )
    parser.add_argument(
        "-d", "--distbound", help="Histogram Euclidean distance bounds", required=True
    )
    parser.add_argument(
        "-dx",
        "--distres",
        help="Histogram Euclidean distance resolution",
        required=True,
    )
    parser.add_argument(
        "-o", type=str, help="Options for CPET", default="./options/options.json"
    )

    args = parser.parse_args()
    options = args.o
    if not os.path.exists(options):
        raise ValueError("Error: Options file not found")
    else:
        with open(options, "r") as f:
            options = json.load(f)

    # Make sure curvature and distance bounds are each lists of 2 numbers, the second greater than the first, convert to floats
    curvbound = args.curvbound.split(",")
    curvbound = [float(i) for i in curvbound]
    distbound = args.distbound.split(",")
    distbound = [float(i) for i in distbound]
    distance_nbins = int(
        (float(distbound[1]) - float(distbound[0])) / float(args.distres)
    )
    curvature_nbins = int(
        (float(curvbound[1]) - float(curvbound[0])) / float(args.curvres)
    )
    if len(curvbound) != 2 or len(distbound) != 2:
        raise ValueError("Error: Curvature and distance bounds must be lists of 2 numbers")
    if float(curvbound[1]) <= float(curvbound[0]) or float(distbound[1]) <= float(
        distbound[0]
    ):
        raise ValueError("Error: Upper bound (right) must be greater than lower bound (left)")

    # Get list of full input topologies and pdbs, sorted
    tops = sorted(f for f in os.listdir(args.tops) if f.endswith(".top"))
    pdbs = sorted(f for f in os.listdir(args.pdbs) if f.endswith(".pdb"))

    # Check to make sure all topologies have a corresponding pdb and the other way around
    for t in tops:
        if t[:-4] + ".pdb" not in pdbs:
            raise ValueError("Error: Topology " + t + " does not have a corresponding pdb")
    for p in pdbs:
        if p[:-4] + ".top" not in tops:
            raise ValueError("Error: Pdb " + p + " does not have a corresponding topology")

    # Get resid list from first pdb
    first_pdb = pdbs[0]
    calc_temp = calculator(options, path_to_pdb=os.path.join(args.pdbs, first_pdb))
    # Residue name and number of every charge. These come from ID, which every filter
    # trims along with x and Q (calc_temp.resids and .residue_number are not trimmed
    # by filter_radius or filter_in_box, so they don't line up with the charges)
    resname_list = [atom[2] for atom in calc_temp.ID]
    resid_list = [atom[3] for atom in calc_temp.ID]

    # Turn resname and resid into a list of dictionaries, one per residue
    residues_list = get_residues(resname_list, resid_list)

    # Save residue list to a file
    with open("residue_list.txt", "w") as f:
        for item in residues_list:
            f.write("%s\n" % item)

    # Now, loop over all pdbs
    chi_squared_breakdown = []
    for pdb in pdbs:
        calc_temp = calculator(options, path_to_pdb=os.path.join(args.pdbs, pdb))
        x_temp = calc_temp.x
        Q_temp = calc_temp.Q
        resname_temp = [atom[2] for atom in calc_temp.ID]
        resid_temp = [atom[3] for atom in calc_temp.ID]
        residues_list_temp = get_residues(resname_temp, resid_temp)

        hist_original = load_hist(pdb, args.tops)
        hist_grid_original = get_hist_grid(
            hist_original, distbound, distance_nbins, curvbound, curvature_nbins
        )

        chi_squared_temp = []

        for residue in residues_list_temp:

            # Find indices where resname_temp is equal to 'resname' and resid_temp is equal to 'resid'
            resname = residue["resname"]
            resid = residue["resid"]
            indices = [
                i
                for i, x in enumerate(resname_temp)
                if x == resname and resid_temp[i] == resid
            ]

            # Remove all lines from x_temp and Q_temp that are not in indices
            x_temp_new = x_temp[indices]
            Q_temp_new = Q_temp[indices]

            calc_temp.x = x_temp_new
            calc_temp.Q = Q_temp_new

            hist = calc_temp.compute_topo_complete_c_shared()
            hist_grid = get_hist_grid(
                hist, distbound, distance_nbins, curvbound, curvature_nbins
            )
            chi_squared_dist = distance_numpy(hist_grid, hist_grid_original)
            print(pdb, residue, chi_squared_dist)
            chi_squared_temp.append([residue, chi_squared_dist])

        chi_squared_breakdown.append([pdb, chi_squared_temp])

    # Get average chi-squared for each residue
    chi_squared_avg = []
    for residue in residues_list:
        chi_squared_residue = []
        for pdb in chi_squared_breakdown:
            for chi_squared in pdb[1]:
                if chi_squared[0] == residue:
                    chi_squared_residue.append(chi_squared[1])
        chi_squared_avg.append(
            [residue, np.mean(chi_squared_residue), np.std(chi_squared_residue)]
        )

    # Sort by mean, and save to file
    chi_squared_avg = sorted(chi_squared_avg, key=lambda x: x[1])
    with open("chi_squared_avg.txt", "w") as f:
        for item in chi_squared_avg:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
