import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import json
import psutil
import time
import os

from sklearn.cluster import AffinityPropagation, HDBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
from CPET.utils.calculator import (
    make_histograms,
    make_histograms_mem,
    make_fields,
    make_5d_tensor,
    construct_distance_matrix,
    construct_distance_matrix_mem,
    construct_distance_matrix_volume,
    construct_distance_matrix_tensor,
    determine_rank,
    reduce_tensor,
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class cluster:
    def __init__(self, options):
        """
        Method to initialize the cluster object
        Takes:
            options: dictionary with options for the cluster object parsed from options.json file
        """
        # General Options
        self.inputpath = options["inputpath"]
        self.outputpath = options["outputpath"]

        # Cluster Specific Options
        if "cluster_method" not in options.keys():
            print("No cluster method specified, defaulting to K-Medoids")
            self.cluster_method = "kmeds"
        else:
            self.cluster_method = options["cluster_method"]
        self.plot_clusters = (
            options["plot_clusters"] if "plot_clusters" in options else False
        )
        self.plot_dwell_times = (
            options["plot_dwell_times"] if "plot_dwell_times" in options else False
        )
        self.cluster_reload = (
            options["cluster_reload"] if "cluster_reload" in options else False
        )
        self.defined_n_clusters = (
            options["defined_n_clusters"] if "defined_n_clusters" in options else None
        )
        self.tensor_threshold = (
            options["tensor_threshold"] if "tensor_threshold" in options else 0.001
        )
        self.rank = options["rank"] if "rank" in options else None
        self.max_rank = options["max_rank"] if "max_rank" in options else 30
        self.min_rank = options["min_rank"] if "min_rank" in options else 1
        # Make sure the provided value for n_clusters and rank is an integer, not a string
        assert self.defined_n_clusters == None or isinstance(
            self.defined_n_clusters, int
        ), "Defined number of clusters must be an integer"
        assert self.rank == None or isinstance(
            self.rank, int
        ), "Rank must be an integer"

        method_dict = {
            "cluster": ["topo_file_list.txt", "top"],
            "cluster_volume": ["field_file_list.txt", "_efield.dat"],
            "cluster_volume_tensor": ["field_file_list.txt", "_efield.dat"],
            "cluster_volume_esp_tensor": ["esp_file_list.txt", "_esp.dat"],
        }
        list_file = method_dict[options["CPET_method"]][0]
        if self.cluster_reload:
            print("Loading distance matrix and topo file list from files!")
            self.file_list = []
            with open(self.outputpath + f"/{list_file}", "r") as file_list:
                for line in file_list:
                    self.file_list.append(line.strip())
            print(
                "{} files found for clustering method {} from input".format(
                    len(self.file_list), options["CPET_method"]
                )
            )
            self.distance_matrix = np.load(self.outputpath + "/distance_matrix.dat.npy")
        else:
            self.file_list = []
            print(method_dict[options["CPET_method"]][1])
            for file in glob(
                self.inputpath + f"/*{method_dict[options['CPET_method']][1]}"
            ):
                self.file_list.append(file)
            if len(self.file_list) == 0:
                print(
                    "No files found for clustering method {}".format(
                        options["CPET_method"]
                    )
                )
                exit()
            self.file_list.sort()
            print(len(self.file_list))
            topo_file_name = self.outputpath + f"/{list_file}"
            with open(topo_file_name, "w") as file_list:
                for i in self.file_list:
                    file_list.write(f"{i} \n")
            print(
                "{} files found for clustering method {}".format(
                    len(self.file_list), options["CPET_method"]
                )
            )
            if options["CPET_method"] == "cluster":
                self.hists = make_histograms(self.file_list)
                self.distance_matrix = construct_distance_matrix(self.hists)
                # self.histlist = make_histograms_mem(self.file_list, self.outputpath)
                # self.distance_matrix = construct_distance_matrix_mem(self.histlist)
            elif options["CPET_method"] == "cluster_volume":
                self.fields = make_fields(self.file_list)
                self.distance_matrix = construct_distance_matrix_volume(self.fields)
            elif options["CPET_method"] == "cluster_volume_tensor":
                self.full_tensor = make_5d_tensor(self.file_list)
                np.save(self.outputpath + "/5d_tensor.npy", self.full_tensor)
                if self.rank == None:
                    self.rank = determine_rank(
                        self.full_tensor, self.tensor_threshold, self.max_rank
                    )  # Time-limiting step (and probably memory)
                self.reduced_tensor, _ = reduce_tensor(self.full_tensor, self.rank)
                self.distance_matrix = construct_distance_matrix_tensor(
                    self.reduced_tensor
                )
            elif options["CPET_method"] == "cluster_volume_esp_tensor":
                # Skip 5d tensor if already made
                if os.path.exists(self.outputpath + "/5d_tensor.npy"):
                    print("5d tensor already found in output directory, skipping")
                    self.full_tensor = np.load(self.outputpath + "/5d_tensor.npy")
                else:
                    self.full_tensor = make_5d_tensor(
                        self.file_list, type="esp"
                    )  # Try to use same fxn as above, to be efficient
                np.save(self.outputpath + "/5d_tensor.npy", self.full_tensor)
                if self.rank == None:
                    self.rank = determine_rank(
                        self.full_tensor,
                        self.tensor_threshold,
                        self.max_rank,
                        self.min_rank,
                    )  # Time-limiting step (and probably memory)
                self.reduced_tensor, self.reconstruction_error = reduce_tensor(
                    self.full_tensor, self.rank
                )
                self.distance_matrix = construct_distance_matrix_tensor(
                    self.reduced_tensor
                )
            np.save(self.outputpath + "/distance_matrix.dat", self.distance_matrix)

    def Cluster(self):
        """
        Run clustering through various methods
        """
        if self.cluster_method == "kmeds":
            self.cluster_results = self.kmeds()
        elif self.cluster_method == "affinity":
            self.cluster_results = self.affinity()
        elif self.cluster_method == "hdbscan":
            self.cluster_results = self.hdbscan()
        else:
            print("Invalid cluster method specified, defaulting to K-Medoids")
            self.cluster_method = "kmeds"
            self.cluster_results = self.kmeds()
        compressed_dictionary = self.cluster_analyze()
        # Save cluster results to json file
        with open(self.outputpath + "/compressed_dictionary.json", "w") as outfile:
            json.dump(compressed_dictionary, outfile, cls=NpEncoder)

    def kmeds(self):
        """
        Method to run K-Medoids clustering, optimized via knee-locator
        Returns:
            cluster_results: dictionary with information about the clusters in best performing K-Medoids
        """
        cluster_results = {}
        distance_matrix = self.distance_matrix
        distance_matrix = distance_matrix**2
        inertia_list = []
        if not self.defined_n_clusters:
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
                f"Using {self.defined_n_clusters} number of clusters with Partitioning around Medoids (PAM)"
            )
            kmeds = KMedoids(
                n_clusters=self.defined_n_clusters,
                random_state=0,
                metric="precomputed",
                method="pam",
                init="k-medoids++",
            )
            kmeds.fit(distance_matrix)
            cluster_results["n_clusters"] = self.defined_n_clusters
        cluster_results["labels"] = list(kmeds.labels_)
        cluster_results["silhouette_score"] = silhouette_score(
            distance_matrix, cluster_results["labels"], metric="precomputed"
        )

        cluster_results["cluster_centers_indices"] = kmeds.medoid_indices_

        return cluster_results

    def affinity(self):
        affinity = AffinityPropagation(
            affinity="precomputed", damping=0.5, max_iter=4000
        )
        affinity_matrix = 1 - self.distance_matrix
        # affinity_matrix[affinity_matrix < 0.2] = 0
        affinity.fit(affinity_matrix)
        self.cluster_results.cluster_centers_indices = affinity.cluster_centers_indices_
        self.cluster_results.labels = list(affinity.labels_)
        self.cluster_results.n_clusters_ = len(
            self.cluster_results.cluster_centers_indices
        )

    def hdbscan(self):
        performance_list = []
        # for percentile_threshold in [70,80,90,99,99.9,99.99,99.999,99.9999,100]:
        for percentile_threshold in [99.9999, 100]:
            threshold = np.percentile(
                self.distance_matrix.flatten(), percentile_threshold
            )
            filtered_distance_matrix = self.distance_matrix.copy()
            filtered_distance_matrix[filtered_distance_matrix > threshold] = 1
            clustering = HDBSCAN(
                min_samples=1,
                min_cluster_size=50,
                store_centers="medoid",
                copy=True,
                allow_single_cluster=False,
                n_jobs=-1,
            )
            clustering.fit(filtered_distance_matrix)
            labels = clustering.labels_
            count_dict = {}
            for i in labels:
                if i in count_dict:
                    count_dict[i] += 1
                else:
                    count_dict[i] = 1
            score = silhouette_score(filtered_distance_matrix, labels)
            performance_list.append(
                [count_dict, score, clustering, threshold, percentile_threshold]
            )
            print(percentile_threshold, score)
            # filter out too noisy convergences
        performance_list_filtered = [
            i for i in performance_list if (i[0].get(-1, float("inf")) <= 250)
        ]
        silhouettes = [i[1] for i in performance_list_filtered]
        if not silhouettes == []:
            best_silhouette = max(silhouettes)
        else:
            print(
                "Significant noisy data (>5%) found here; take these results with a grain of salt"
            )
            performance_list_filtered = performance_list
            silhouettes = [i[1] for i in performance_list_filtered]
            best_silhouette = max(silhouettes)
        best_performance = performance_list_filtered[
            len(silhouettes) - 1 - silhouettes[::-1].index(best_silhouette)
        ]
        labels = best_performance[2].labels_
        best_filtered_matrix = self.cluster_results.distance_matrix.copy()
        best_filtered_matrix[best_filtered_matrix > best_performance[3]] = 1
        cluster_centers_indices = [
            np.where(
                np.all(best_filtered_matrix == best_performance[2].medoids_[i], axis=1)
            )[0][0]
            for i in range(len(best_performance[2].medoids_))
        ]
        # Define number of clusters to include the 'noise' cluster
        n_clusters_ = len(best_performance[0].keys())
        # If noise cluster exists, add a null cluster center index
        if not n_clusters_ == len(best_performance[2].medoids_):
            cluster_centers_indices = [""] + cluster_centers_indices

        print(
            f"Best performance with {best_performance[4]}% cutoff and silhouette score of {best_silhouette}"
        )

    def cluster_analyze(self):
        """
        Method to analyze, format, and plot clustering results
        Takes:
            self: information about clustering and topology files
        Returns:
            compressed_dictionary: dictionary with information about the clusters
        """
        # generate compressed distance matrix of cluster centers
        self.reduced_distance_matrix = self.distance_matrix[
            self.cluster_results["cluster_centers_indices"]
        ][:, self.cluster_results["cluster_centers_indices"]]

        compressed_dictionary = {}
        # get count of a value in a list
        for i in range(self.cluster_results["n_clusters"]):
            temp_dict = {}
            temp_dict["count"] = list(self.cluster_results["labels"]).count(i)
            temp_dict["index_center"] = self.cluster_results["cluster_centers_indices"][
                i
            ]
            temp_dict["name_center"] = self.file_list[temp_dict["index_center"]]
            temp_dict["percentage"] = (
                float(temp_dict["count"])
                / float(len(self.cluster_results["labels"]))
                * 100
            )
            cluster_indices = [
                y for y, x in enumerate(self.cluster_results["labels"]) if x == i
            ]
            temp_dict["mean_distance"] = np.mean(
                self.distance_matrix[temp_dict["index_center"]][cluster_indices]
            )
            temp_dict["max_distance"] = np.max(
                self.distance_matrix[temp_dict["index_center"]][cluster_indices]
            )
            temp_zip = zip(
                [self.file_list[i].split("/")[-1] for i in cluster_indices],
                [
                    self.distance_matrix[temp_dict["index_center"]][i]
                    for i in cluster_indices
                ],
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
        print(f"Silhouette Score: {self.cluster_results['silhouette_score']}")
        # compressed_dictionary["boundary_inds"] = self.cluster_results["bounary_list_inds"]
        compressed_dictionary["silhouette"] = self.cluster_results["silhouette_score"]
        compressed_dictionary["n_clusters"] = self.cluster_results["n_clusters"]
        compressed_dictionary["total_count"] = len(self.cluster_results["labels"])

        if self.plot_clusters == True:
            # Plot clusters with Multi-Dimensional Scaling
            mds = MDS(n_components=3, dissimilarity="precomputed", random_state=0)
            projection = mds.fit_transform(
                self.distance_matrix
            )  # Directly feed the distance matrix
            color_palette = sns.color_palette("deep", 12)
            cluster_colors = [
                color_palette[label] for label in self.cluster_results["labels"]
            ]

            # Define different perspectives
            perspectives = [
                (30, 30),  # Elevation=30°, Azimuth=30°
                (30, 120),  # Elevation=30°, Azimuth=120°
                (30, 210),  # Elevation=30°, Azimuth=210°
                (30, 300),  # Elevation=30°, Azimuth=300°
                (90, 0),  # Top view
            ]

            # Create a 3D scatter plot from different perspectives
            for elev, azim in perspectives:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    projection[:, 0],
                    projection[:, 1],
                    projection[:, 2],
                    s=10,
                    linewidth=0,
                    alpha=0.25,
                    c=cluster_colors,
                )
                ax.view_init(elev, azim)
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
                plt.title(f"View from elevation {elev}°, azimuth {azim}°")
                plt.show()
        if self.plot_dwell_times == True:
            print(0)
        return compressed_dictionary
