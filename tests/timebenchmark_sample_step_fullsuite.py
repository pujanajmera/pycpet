import os
import numpy as np
import warnings
from glob import glob
from random import choice
from CPET.source.calculator import calculator
from CPET.utils.calculator import make_histograms, construct_distance_matrix
from CPET.source.benchmark import gen_param_dist_mat
import json
import argparse
import matplotlib.pyplot as plt
from CPET.source.CPET import CPET


def main():
    parser = argparse.ArgumentParser(description='CPET: A tool for computing and analyzing electric fields in proteins')
    parser.add_argument('-o', type=json.loads, help='Options for CPET', default=json.load(open("./options/options.json")))
    args = parser.parse_args()
    options = args.o
    cpet = CPET(options)
    files_input = glob(cpet.inputpath + "/*.pdb")
    if len(files_input) != 1:
        print("Less or more than 1 pdb file found in directory, exiting")
        exit()
    topo_files = []
    benchmark_step_sizes = [0.1,0.05,0.01,0.005,0.001]
    benchmark_samples = [100000,50000,10000,5000,1000]
    benchmark_radii = [80,70,60,50,40,30,20,10]
    #benchmark_step_sizes = [0.001]
    #benchmark_samples = [100000]
    #benchmark_radii = [40,30,20]

    for step_size in benchmark_step_sizes:
        for n_samples in benchmark_samples:
            for radii in benchmark_radii:
                for file in files_input:
                    files_done = [x for x in os.listdir(cpet.outputpath) if x.split(".")[-1]=="top"]
                    protein = file.split("/")[-1].split(".")[0]
                    outstring = "{}_{}_{}_{}.top".format(protein, n_samples, str(step_size)[2:], radii)
                    if outstring in files_done:
                        topo_files.append(cpet.outputpath + "/" + outstring)
                        continue
                    cpet.options["n_samples"] = n_samples
                    cpet.options["step_size"] = step_size
                    if radii is not None:
                        cpet.options["filter_radius"] = radii
                    cpet.calculator = calculator(cpet.options, path_to_pdb = file)
                    if cpet.m == "topo_GPU":
                        hist = cpet.calculator.compute_topo_GPU_batch_filter()
                    else:
                        hist = cpet.calculator.compute_topo_complete_c_shared()
                    np.savetxt(cpet.outputpath + "/" + outstring, hist)
                    topo_files.append(cpet.outputpath + "/" + outstring)

main()
