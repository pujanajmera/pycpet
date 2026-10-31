import matplotlib.pyplot as plt
import numpy as np
import json
import os
import logging

from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from kneed import KneeLocator
from glob import glob
from CPET.utils.calculator import (
    make_histograms,
    make_fields,
    make_5d_tensor,
    construct_distance_matrix,
    construct_distance_matrix_volume,
    construct_distance_matrix_tensor,
    determine_rank,
    reduce_tensor,
)
from CPET.utils.io import default_options_initializer

log = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """
    JSON encoder that also writes NumPy integers, floats and arrays.
    """
    def default(self, obj):
        """
        Convert a NumPy value to its Python equivalent for JSON.

        Parameters
        ----------
        obj : object
            The value to convert.

        Returns
        -------
        object
            An ``int``, ``float`` or ``list`` for NumPy values; other types go to the
            standard encoder.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class cluster:
    """
    Cluster a set of topologies, field volumes or electrostatic potential volumes.

    Creating the object finds the input files (``.top``, ``_efield.dat`` or
    ``_esp.dat``, depending on ``CPET_method``) and computes the distance matrix
    between them, saving the file list and matrix to ``outputpath``. With
    ``cluster_reload``, it loads those saved files instead. Call :meth:`Cluster`
    to cluster.

    Parameters
    ----------
    options : dict
        Options. Reads ``CPET_method``, ``inputpath``, ``outputpath`` and the
        clustering options (see the Options Reference).
    """
    def __init__(self, options):
        """
        Find the input files and compute or load the distance matrix.
        """
        # Fill defaults in case cluster is used outside of CPET.py
        options = default_options_initializer(options)

        # General Options
        self.inputpath = options["inputpath"]
        self.outputpath = options["outputpath"]

        # Cluster Specific Options
        if "cluster_method" not in options:
            log.info("No cluster method specified, defaulting to K-Medoids")
            self.cluster_method = "kmeds"
        else:
            self.cluster_method = options["cluster_method"]
        self.plot_clusters = options["plot_clusters"]
        self.plot_dwell_times = options["plot_dwell_times"]
        self.cluster_reload = options["cluster_reload"]
        self.defined_n_clusters = options["defined_n_clusters"]
        self.tensor_threshold = options["tensor_threshold"]
        self.rank = options["rank"]
        self.max_rank = options["max_rank"]
        self.min_rank = options["min_rank"]
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
            log.info("Loading saved distance matrix and file list")
            self.file_list = []
            with open(self.outputpath + f"/{list_file}", "r") as file_list:
                for line in file_list:
                    self.file_list.append(line.strip())
            log.info(
                "Found {} files in saved file list for clustering method {}".format(
                    len(self.file_list), options["CPET_method"]
                )
            )
            self.distance_matrix = np.load(self.outputpath + "/distance_matrix.dat.npy")
        else:
            self.file_list = []
            log.debug(
                f"Looking for files ending in: {method_dict[options['CPET_method']][1]}"
            )
            for file in glob(
                self.inputpath + f"/*{method_dict[options['CPET_method']][1]}"
            ):
                self.file_list.append(file)
            if len(self.file_list) == 0:
                log.error(
                    "No files found for clustering method {}".format(
                        options["CPET_method"]
                    )
                )
                exit()
            self.file_list.sort()
            topo_file_name = self.outputpath + f"/{list_file}"
            with open(topo_file_name, "w") as file_list:
                for i in self.file_list:
                    file_list.write(f"{i} \n")
            log.info(
                "Found {} files for clustering method {}".format(
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
                    log.info("5D tensor already found in output directory, skipping")
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
        Cluster with the chosen method and save a summary of the results.

        Runs ``cluster_method`` (K-Medoids unless another is set), then writes the
        summary from :meth:`cluster_analyze` to ``outputpath/compressed_dictionary.json``.
        """
        if self.cluster_method == "kmeds":
            self.cluster_results = self.kmeds()
        elif self.cluster_method == "affinity":
            self.cluster_results = self.affinity()
        elif self.cluster_method == "hdbscan":
            self.cluster_results = self.hdbscan()
        else:
            log.warning("Invalid cluster method specified, defaulting to K-Medoids")
            self.cluster_method = "kmeds"
            self.cluster_results = self.kmeds()
        compressed_dictionary = self.cluster_analyze()
        # Save cluster results to json file
        with open(self.outputpath + "/compressed_dictionary.json", "w") as outfile:
            json.dump(compressed_dictionary, outfile, cls=NpEncoder)

    def kmeds(self):
        """
        Cluster with K-Medoids (PAM) on the squared distance matrix.

        Uses ``defined_n_clusters`` clusters if set. Otherwise fits 1-50 clusters
        and picks the elbow of the inertia curve.

        Returns
        -------
        cluster_results : dict
            ``n_clusters``, ``labels`` (cluster of each file), ``silhouette_score``
            and ``cluster_centers_indices`` (index of each cluster's medoid).
        """
        from sklearn_extra.cluster import KMedoids

        cluster_results = {}
        distance_matrix = self.distance_matrix
        distance_matrix = distance_matrix ** 2
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
                log.info(f"K-Medoids with {i + 1} clusters, inertia: {kmeds.inertia_}")
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

            log.info(
                f"Using {kn.elbow} clusters with Partitioning Around Medoids (PAM), chosen by the elbow method"
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
            log.info(
                f"Using {self.defined_n_clusters} clusters with Partitioning Around Medoids (PAM)"
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
        """
        Cluster with affinity propagation, using 1 - distance as the affinity.

        Notes
        -----
        Not working yet: it stores results on ``self.cluster_results`` before that
        exists and returns nothing, so ``cluster_method: "affinity"`` fails.
        """
        from sklearn.cluster import AffinityPropagation

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
        """
        Cluster with HDBSCAN, trying two distance cutoffs and keeping the best
        silhouette score.

        Notes
        -----
        Not working yet: it reads ``self.cluster_results`` before that exists and
        returns nothing, so ``cluster_method: "hdbscan"`` fails.
        """
        from sklearn.cluster import HDBSCAN

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
            log.info(f"Percentile threshold {percentile_threshold}: silhouette score {score}")
            # filter out too noisy convergences
        performance_list_filtered = [
            i for i in performance_list if (i[0].get(-1, float("inf")) <= 250)
        ]
        silhouettes = [i[1] for i in performance_list_filtered]
        if not silhouettes == []:
            best_silhouette = max(silhouettes)
        else:
            log.warning(
                "Significant noise in data (>5%), interpret these results with caution"
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

        log.info(
            f"Best performance with {best_performance[4]}% cutoff and silhouette score of {best_silhouette}"
        )

    def cluster_analyze(self):
        """
        Summarize the clusters, and optionally plot them.

        Returns
        -------
        compressed_dictionary : dict
            One entry per cluster, keyed by cluster number and sorted from largest to
            smallest, with the ``count``, ``percentage``, the medoid's index
            (``index_center``) and file (``name_center``), ``mean_distance`` and
            ``max_distance`` from the medoid, and the cluster's ``files`` and their
            ``distances`` from the medoid. Also has ``silhouette``, ``n_clusters`` and
            ``total_count``.
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
        log.info("Percentage of each cluster:")
        for key in compressed_dictionary:
            if type(key) == int:
                log.info(
                    f"Cluster {key}: {compressed_dictionary[key]['percentage']}% of total"
                )
            else:
                if key.isnumeric():
                    log.info(
                        f"Cluster {key}: {compressed_dictionary[key]['percentage']}% of total"
                    )
        log.info(f"Silhouette score: {self.cluster_results['silhouette_score']}")
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
            color_palette = plt.get_cmap("tab20").colors
            cluster_colors = [
                (0.5, 0.5, 0.5)
                if label == -1
                else color_palette[int(label) % len(color_palette)]
                for label in self.cluster_results["labels"]
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
            log.warning("Dwell time plotting (plot_dwell_times) is not implemented yet, skipping")
        return compressed_dictionary
