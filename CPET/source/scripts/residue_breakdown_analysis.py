from CPET.source.calculator import calculator
import argparse
import os
import json


def main():
    #Parse arguments
    parser = argparse.ArgumentParser(description='Residue breakdown analysis')
    parser.add_argument('-t', '--tops', help='Topology file directory', required=True)
    parser.add_argument('-p', '--pdbs', help='PDB file directory', required=True)
    parser.add_argument('-c', '--curvbound', help='Histogram curvature bounds', required=True)
    parser.add_argument('-d', '--distbound', help='Histogram Euclidean distance bounds', required=True)
    parser.add_argument("-o", type=str, help="Options for CPET", default="./options/options.json")

    args = parser.parse_args()
    options = args.o
    if not os.path.exists(options):
        ValueError('Error: Options file not found')
    else:
        with open(options, "r") as f:
            options = json.load(f)

    #Make sure curvature and distance bounds are each lists of 2 numbers, the second greater than the first
    curvbound = args.curvbound.split(',')
    distbound = args.distbound.split(',')
    if len(curvbound) != 2 or len(distbound) != 2:
        ValueError('Error: Curvature and distance bounds must be lists of 2 numbers')
    if float(curvbound[1]) <= float(curvbound[0]) or float(distbound[1]) <= float(distbound[0]):
        ValueError('Error: Second bound must be greater than first bound')

    #Get list of full input topologies and pdbs, sorted
    tops = os.listdir(args.tops)
    tops = sorted(tops)
    pdbs = os.listdir(args.pdbs)
    pdbs = sorted(pdbs)

    #Check to make sure all topologies have a corresponding pdb (topologies end in .top, pdbs end in .pdb)
    for t in tops:
        if t[:-4] + '.pdb' not in pdbs:
            ValueError('Error: Topology ' + t + ' does not have a corresponding pdb')
    
    #Get resid list from first pdb
    first_pdb = pdbs[0]
    calc_temp = calculator(options, path_to_pdb=first_pdb)
    resid_list = calc_temp.resids #List of numbers, only has non-zeroed resids from the options file (e.g. zeroing active site)

    #Save resid list as a file
    with open('resid_list.txt', 'w') as f:
        for resid in resid_list:
            f.write(str(resid) + '\n')

    #Loop over all pdbs

    #For each pdb, get resid list but save just the first pdb resid list
    #After getting resid list, zero all charges but the residue of interest
    #Run CPET
    #Use histogram bounds from prior run
    #Generate histograms, compute chi-squared distance from topology to the frame topology (previously computed)
    #After looping over all pdbs, show, for each residue, the distance between the resid zeroed topology and the frame topology over the MD
    #Rank in order of most varying or closest, or both

if __name__ == '__main__':
    main()