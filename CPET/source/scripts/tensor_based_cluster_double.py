from CPET.utils.calculator import (
    make_5d_tensor,
    construct_distance_matrix_tensor,
    determine_rank,
    reduce_tensor,
)
from CPET.source.cluster import NpEncoder

import argparse
import os
import json
import numpy as np
from glob import glob

# Import clustering modules
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from scipy.spatial.distance import pdist, squareform


def kmeds(distance_matrix, defined_n_clusters=None):
    """
    Method to run K-Medoids clustering, optimized via knee-locator
    Returns:
        cluster_results: dictionary with information about the clusters in best performing K-Medoids
    """
    cluster_results = {}
    distance_matrix = distance_matrix
    distance_matrix = distance_matrix**2
    inertia_list = []
    if not defined_n_clusters:
        for i in range(50):
            kmeds = KMedoids(
                n_clusters=i + 1,
                random_state=0,
                metric="precomputed",
                method="pam",
                init="k-medoids++",
            )
            kmeds.fit(distance_matrix)
            labels = list(kmeds.labels_)
            inertia_list.append(kmeds.inertia_)
            print(i + 1, kmeds.inertia_)
        # Use second-derivate based elbow locating with 1-15 clusters
        kn = KneeLocator(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
            ],
            inertia_list,
            curve="convex",
            direction="decreasing",
        )

        print(
            f"Using {kn.elbow} number of clusters with Partitioning around Medoids (PAM), derived from elbow method"
        )
        cluster_results["n_clusters"] = int(kn.elbow)
        kmeds = KMedoids(
            n_clusters=kn.elbow,
            random_state=0,
            metric="precomputed",
            method="pam",
            init="k-medoids++",
        )

        kmeds.fit(distance_matrix)
    else:
        print(
            f"Using {defined_n_clusters} number of clusters with Partitioning around Medoids (PAM)"
        )
        kmeds = KMedoids(
            n_clusters=defined_n_clusters,
            random_state=0,
            metric="precomputed",
            method="pam",
            init="k-medoids++",
        )
        kmeds.fit(distance_matrix)
        cluster_results["n_clusters"] = defined_n_clusters
    cluster_results["labels"] = list(kmeds.labels_)
    cluster_results["silhouette_score"] = silhouette_score(
        distance_matrix, cluster_results["labels"], metric="precomputed"
    )

    cluster_results["cluster_centers_indices"] = kmeds.medoid_indices_

    return cluster_results


def cluster_analyze(distance_matrix, cluster_results, file_list):
    """
    Method to analyze, format, and plot clustering results
    Takes:
        self: information about clustering and topology files
    Returns:
        compressed_dictionary: dictionary with information about the clusters
    """
    # generate compressed distance matrix of cluster centers
    reduced_distance_matrix = distance_matrix[
        cluster_results["cluster_centers_indices"]
    ][:, cluster_results["cluster_centers_indices"]]

    compressed_dictionary = {}
    # get count of a value in a list
    for i in range(cluster_results["n_clusters"]):
        temp_dict = {}
        temp_dict["count"] = list(cluster_results["labels"]).count(i)
        temp_dict["index_center"] = cluster_results["cluster_centers_indices"][i]
        temp_dict["name_center"] = file_list[temp_dict["index_center"]]
        temp_dict["percentage"] = (
            float(temp_dict["count"]) / float(len(cluster_results["labels"])) * 100
        )
        cluster_indices = [y for y, x in enumerate(cluster_results["labels"]) if x == i]
        temp_dict["mean_distance"] = np.mean(
            distance_matrix[temp_dict["index_center"]][cluster_indices]
        )
        temp_dict["max_distance"] = np.max(
            distance_matrix[temp_dict["index_center"]][cluster_indices]
        )
        temp_zip = zip(
            [file_list[i].split("/")[-1] for i in cluster_indices],
            [distance_matrix[temp_dict["index_center"]][i] for i in cluster_indices],
        )
        sorted_temp_zip = sorted(temp_zip, key=lambda x: x[1])
        temp_dict["files"], temp_dict["distances"] = zip(*sorted_temp_zip)
        compressed_dictionary[str(i)] = temp_dict

    # resort by count
    compressed_dictionary = dict(
        sorted(
            compressed_dictionary.items(),
            key=lambda item: int(item[1]["count"]),
            reverse=True,
        )
    )
    # print percentage of each cluster
    print("Percentage of each cluster: ")
    for key in compressed_dictionary.keys():
        if type(key) == int:
            print(
                f"Cluster {key}: {compressed_dictionary[key]['percentage']}% of total"
            )
        else:
            if key.isnumeric():
                print(
                    f"Cluster {key}: {compressed_dictionary[key]['percentage']}% of total"
                )
    print(f"Silhouette Score: {cluster_results['silhouette_score']}")
    # compressed_dictionary["boundary_inds"] = self.cluster_results["bounary_list_inds"]
    compressed_dictionary["silhouette"] = cluster_results["silhouette_score"]
    compressed_dictionary["n_clusters"] = cluster_results["n_clusters"]
    compressed_dictionary["total_count"] = len(cluster_results["labels"])
    return compressed_dictionary


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Two-site tensor-based clustering.\nThis is not a restartable script, and will repeat everything from the beginning"
    )
    parser.add_argument(
        "-d1", "--dir1", help="First directory with fields", required=True
    )
    parser.add_argument(
        "-d2", "--dir2", help="Second directory with fields", required=True
    )
    parser.add_argument("-u", "--outputpath", help="Output path", required=True)
    parser.add_argument(
        "-o", type=str, help="Options for CPET", default="./options/options.json"
    )

    args = parser.parse_args()
    options = args.o
    if not os.path.exists(options):
        ValueError("Error: Options file not found")
    else:
        with open(options, "r") as f:
            options = json.load(f)

    defined_n_clusters = (
        options["defined_n_clusters"] if "defined_n_clusters" in options else None
    )
    tensor_threshold = (
        options["tensor_threshold"] if "tensor_threshold" in options else 0.001
    )
    rank = options["rank"] if "rank" in options else None
    max_rank = options["max_rank"] if "max_rank" in options else 30

    outputpath = args.outputpath
    if not os.path.exists(outputpath):
        # Make directory
        os.makedirs(outputpath)

    # Make sure file names in both directories are identical
    dir1_files = os.listdir(args.dir1).sort()
    dir2_files = os.listdir(args.dir2).sort()
    if dir1_files != dir2_files:
        ValueError("Error: Files in directories are not identical")

    # Parse first directory:
    inputpath = args.dir1
    file_list1 = []
    for file in glob(inputpath + f"/*_esp.dat"):
        file_list1.append(file)
    file_list1.sort()
    list_file = "esp_file_list1.txt"
    topo_file_name = outputpath + f"/{list_file}"
    with open(topo_file_name, "w") as file_list:
        for i in file_list1:
            file_list.write(f"{i} \n")
    full_tensor = make_5d_tensor(file_list1, type="esp")
    if rank == None:
        rank1 = determine_rank(
            full_tensor, tensor_threshold, max_rank
        )  # Time-limiting step (and probably memory)
    else:
        rank1 = rank
    reduced_tensor1, reconstruction_error1 = reduce_tensor(full_tensor, rank1)
    factor_mat1 = reduced_tensor1.factors[0]
    print(f"Reconstruction error for first tensor: {reconstruction_error1}")
    np.save(outputpath + "/factor_mat1.npy", factor_mat1)  # Shape (L,N)

    # Parse second directory, using first directory file names as reference:
    inputpath = args.dir2
    file_list2 = []
    for file in glob(inputpath + f"/*_esp.dat"):
        file_list2.append(file)
    file_list2.sort()
    list_file = "esp_file_list2.txt"
    topo_file_name = outputpath + f"/{list_file}"
    with open(topo_file_name, "w") as file_list:
        for i in file_list2:
            file_list.write(f"{i} \n")
    full_tensor = make_5d_tensor(file_list2, type="esp")
    if rank == None:
        rank2 = determine_rank(full_tensor, tensor_threshold, max_rank)
    else:
        rank2 = rank
    reduced_tensor2, reconstruction_error2 = reduce_tensor(full_tensor, rank2)
    factor_mat2 = reduced_tensor2.factors[0]
    print(f"Reconstruction error for second tensor: {reconstruction_error2}")
    np.save(outputpath + "/factor_mat2.npy", factor_mat2)  # Shape (M,N)

    combined_file_list = file_list1  # Both should be identical, since its combined
    with open(outputpath + "/combined_file_list.txt", "w") as file_list:
        for i in combined_file_list:
            file_list.write(f"{i} \n")

    # Combine reduced tensors to make shape (L+M, N)
    print(f"Factor matrix 1 shape: {factor_mat1.shape}")
    print(f"Factor matrix 2 shape: {factor_mat2.shape}")
    combined_fac_mat = np.concatenate((factor_mat1, factor_mat2), axis=1)
    np.save(outputpath + "/combined_factor_mat.npy", combined_fac_mat)
    print(f"Combined factor matrix shape: {combined_fac_mat.shape}")
    # Construct distance matrix
    distance_matrix = squareform(pdist(combined_fac_mat, metric="euclidean"))
    np.save(outputpath + "/distance_matrix_combined.npy", distance_matrix)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    # Run K-Medoids clustering
    cluster_results = kmeds(distance_matrix, defined_n_clusters)

    compressed_dictionary = cluster_analyze(
        distance_matrix, cluster_results, combined_file_list
    )
    # Save cluster results to json file
    with open(outputpath + "/compressed_dictionary.json", "w") as outfile:
        json.dump(compressed_dictionary, outfile, cls=NpEncoder)


if __name__ == "__main__":
    main()
