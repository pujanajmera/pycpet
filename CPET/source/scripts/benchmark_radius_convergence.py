import numpy as np
from CPET.source.calculator import calculator
from CPET.utils.calculator import make_histograms, construct_distance_matrix
from CPET.source.CPET import CPET
import warnings

warnings.filterwarnings(action="ignore")
import json
import argparse
from random import choice
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob


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
    if "benchmark_radii" in options:
        benchmark_radii = options["benchmark_radii"]
    else:
        print(
            "No benchmark radii specified, using default values: [None,70,60,50,40,30,20]"
        )
        benchmark_radii = [None, 70, 60, 50, 40, 30, 20, 10]

    for radius in benchmark_radii:
        print(f"Running for radius: {radius}")
        for i in range(3):
            for file in files_input:
                files_done = [
                    x for x in os.listdir(cpet.outputpath) if x.split(".")[-1] == "top"
                ]
                protein = file.split("/")[-1].split(".")[0]
                outstring = "{}_{}_{}.top".format(protein, radius, i)
                if outstring in files_done:
                    topo_files.append(cpet.outputpath + "/" + outstring)
                    continue
                if radius is not None:
                    cpet.options["filter_radius"] = radius
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
        if len(topo_file_protein) != num * len(benchmark_radii):
            raise ValueError(
                "Incorrect number of output topologies for requested benchmark parameters"
            )

        histograms = make_histograms(topo_file_protein, plot=False)
        distance_matrix = construct_distance_matrix(histograms)

        distances = pd.DataFrame(distance_matrix)

        # Modify file names
        name = (
            topo_file_protein[0].split("/")[-1].split("_")[0]
            + "_"
            + topo_file_protein[0].split("/")[-1].split("_")[1]
            + "_"
            + topo_file_protein[0].split("/")[-1].split("_")[2]
            + "_"
        )
        labels = topo_file_protein
        labels = [
            label.replace(".top", "").split("/")[-1].replace(name, "")
            for label in labels
        ]
        print(labels)

        # Map each label to its group
        group_map = {label: label.split("_")[0] for label in labels}
        grouped_labels = [group_map[label] for label in labels]
        print(group_map)
        print(grouped_labels)
        # Apply the new labels to the DataFrame
        distances.columns = grouped_labels
        distances.index = grouped_labels

        # Aggregate by taking the mean within each group for both rows and columns
        grouped = distances.groupby(level=0).mean()
        averaged_distances = grouped.T.groupby(level=0).mean()

        # Ensure the matrix is symmetric
        averaged_distances = (averaged_distances + averaged_distances.T) / 2

        # (Optional) Plot the distance matrix
        # First, close any open plots
        plt.close()
        plt.figure(figsize=(10, 8))
        sns.heatmap(averaged_distances, cmap="Greens_r", annot=True, linewidths=0.1)
        plt.title("Averaged Distance Matrix")
        plt.show()
        plt.imsave(
            f"averaged_distance_matrix_{file.split('/')[-1].split('.')[0]}.png",
            averaged_distances,
        )


main()
