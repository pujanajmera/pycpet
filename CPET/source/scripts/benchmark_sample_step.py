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
    parser = argparse.ArgumentParser(
        description="CPET: A tool for computing and analyzing electric fields in proteins"
    )
    parser.add_argument(
        "-o",
        type=json.loads,
        help="Options for CPET",
        default=json.load(open("./options/options.json")),
    )
    args = parser.parse_args()
    options = args.o
    cpet = CPET(options)
    files_input = glob(cpet.inputpath + "/*.pdb")
    num = 3
    if len(files_input) < 3:
        warnings.warn(
            "Less than 3 pdb files found in the input directory, benchmarking on {} files. This may be insufficient sampling".format(
                len(files_input)
            )
        )
        num = len(files_input)
    if len(files_input) > 3:
        warnings.warn(
            "More than 3 pdb files found in the input directory, choosing 3 random pdbs to benchmarking on"
        )
        files_input = [choice(files_input) for i in range(num)]
    topo_files = []
    benchmark_step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    benchmark_samples = [500000, 100000, 50000, 10000, 5000, 1000]
    for step_size in benchmark_step_sizes:
        for n_samples in benchmark_samples:
            for i in range(3):
                for file in files_input:
                    files_done = [
                        x
                        for x in os.listdir(cpet.outputpath)
                        if x.split(".")[-1] == "top"
                    ]
                    protein = file.split("/")[-1].split(".")[0]
                    outstring = "{}_{}_{}_{}.top".format(
                        protein, n_samples, str(step_size)[2:], i
                    )
                    if outstring in files_done:
                        topo_files.append(cpet.outputpath + "/" + outstring)
                        continue
                    cpet.options["n_samples"] = n_samples
                    cpet.options["step_size"] = step_size
                    cpet.calculator = calculator(cpet.options, path_to_pdb=file)
                    if cpet.m == "topo_GPU":
                        hist = cpet.calculator.compute_topo_GPU_batch_filter()
                    else:
                        hist = cpet.calculator.compute_topo_complete_c_shared()
                    np.savetxt(cpet.outputpath + "/" + outstring, hist)
                    topo_files.append(cpet.outputpath + "/" + outstring)
    for file in files_input:
        topo_file_protein = [
            x for x in topo_files if file.split("/")[-1].split(".")[0] in x
        ]
        if len(topo_file_protein) != num * len(benchmark_samples) * len(
            benchmark_step_sizes
        ):
            raise ValueError(
                "Incorrect number of output topologies for requested benchmark parameters"
            )
        histograms = make_histograms(topo_file_protein, plot=False)
        plt.close()
        distance_matrix = construct_distance_matrix(histograms)
        avg_dist = gen_param_dist_mat(distance_matrix, topo_file_protein)


main()
