import numpy as np
from CPET.source.calculator import calculator
from CPET.utils.calculator import make_histograms, construct_distance_matrix_alt2, construct_distance_matrix
import warnings 
warnings.filterwarnings(action='ignore')
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='CPET: A tool for computing and analyzing electric fields in proteins')
    parser.add_argument('-o', type=str, help='Options for CPET', default="./options/options.json")
    parser.add_argument('-pdb', type=str, help='Path to pdb file', default="./tests/1a0m.pdb")
    args = parser.parse_args()
    options = args.o
    options = json.load(open(options))
    file = args.pdb

    topo_file_list = []

    iter = 3

    current_dir_files = os.listdir()

    for radius in [None,90,80,70,60,50,40,30,20]:
        print(f"Running for radius: {radius}")
        if radius is not None:
            options["filter_radius"] = radius
        for i in range(iter):
            filestring = f"rad_conv_{radius}_{i}.top"
            if filestring in current_dir_files:
                print(filestring+" already exists. Skipping...")
                topo_file_list.append(filestring)
                continue
            topo = calculator(options, path_to_pdb = file)
            ret = topo.compute_topo_GPU_batch_filter()
            np.savetxt(filestring, ret)
            topo_file_list.append(filestring)
    
    histograms = make_histograms(topo_file_list,plot=False)
    distance_matrix = construct_distance_matrix(histograms)

    distances = pd.DataFrame(distance_matrix)

    #Modify file names

    labels = topo_file_list
    labels = [label.replace(".top","").split("/")[-1].replace("rad_conv_","") for label in labels]

    # Map each label to its group
    group_map = {label: label.split('_')[0] for label in labels}
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
    plt.figure(figsize=(10,8))
    sns.heatmap(averaged_distances, cmap="Greens_r", annot=True,linewidths=0.1)
    plt.title("Averaged Distance Matrix")
    plt.show()
    plt.imsave("averaged_distance_matrix.png",averaged_distances)
    
main()
