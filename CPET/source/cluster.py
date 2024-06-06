import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx
from sklearn_extra.cluster import KMedoids
from glob import glob
from CPET.utils.calculator import (
    make_histograms,
    construct_distance_matrix,
    construct_distance_matrix_alt,
    construct_distance_matrix_alt2,
    construct_distance_matrix_volume,
    make_fields,
)


class cluster:
    def __init__(self, options):
        self.cluster_method = options["cluster_method"]
        self.cluster_reload = (
            options["cluster_reload"] if "cluster_reload" in options else False
        )
        self.inputpath = options["inputpath"]
        self.outputpath = options["outputpath"]

        if options["CPET_method"] == "cluster":
            self.topo_file_list = []
            for file in glob(self.inputpath + "/*.top"):
                self.topo_file_list.append(file)
            self.topo_file_list.sort()
            topo_file_name = self.outputpath + "/topo_file_list.txt"
            with open(topo_file_name, "w") as file_list:
                for i in self.topo_file_list:
                    file_list.write(f"{i} \n")
            print("{} files found for clustering".format(len(self.topo_file_list)))
            self.hists = make_histograms(self.topo_file_list)
            if self.cluster_reload:
                print("Loading distance matrix from file!")
                self.distance_matrix = np.load(self.outputpath + "/distance_matrix.dat")
            else:
                self.distance_matrix = construct_distance_matrix_alt2(self.hists)
                np.save(self.outputpath + "/distance_matrix.dat", self.distance_matrix)
        elif options["CPET_method"] == "cluster_volume":
            self.field_file_list = []
            for file in glob(self.inputpath + "/*_efield.dat"):
                self.field_file_list.append(file)
            self.field_file_list.sort()
            field_file_name = self.outputpath + "/field_file_list.txt"
            with open(field_file_name, "w") as file_list:
                for i in self.field_file_list:
                    file_list.write(f"{i} \n")
            print("{} files found for clustering".format(len(self.field_file_list)))
            self.fields = make_fields(self.field_file_list)
            if self.cluster_reload:
                print("Loading distance matrix from file!")
                self.distance_matrix = np.load(self.outputpath + "/distance_matrix.dat")
            else:
                self.distance_matrix = construct_distance_matrix_volume(self.fields)
                np.save(self.outputpath + "/distance_matrix.dat", self.distance_matrix)

    def Cluster(self):
        if self.cluster_method == "affinity":
            self.cluster_results = self.affinity()
        elif self.cluster_method == "kmeds":
            self.cluster_results = self.kmeds()
        elif self.cluster_method == "hdbscan":
            self.cluster_results = self.hdbscan()

        self.compress()

    def kmeds(self):
        self.cluster_results = {}
        distance_matrix = self.distance_matrix
        silhouette_list = []
        for i in range(6):
            kmeds = KMedoids(
                n_clusters=i + 2, random_state=0, metric="precomputed", method="pam"
            )
            kmeds.fit(distance_matrix)
            labels = list(kmeds.labels_)
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            print(i + 2, score)
            silhouette_list.append(score)
        max_index = silhouette_list.index(max(silhouette_list))
        print(
            f"Using {max_index+2} number of clusters with Partitioning around Medoids (PAM)"
        )
        kmeds = KMedoids(
            n_clusters=max_index + 2, random_state=0, metric="precomputed", method="pam"
        )
        kmeds.fit(distance_matrix)
        self.cluster_results["labels"] = list(kmeds.labels_)
        self.cluster_results["n_clusters"] = max_index + 2
        self.cluster_results["cluster_centers_indices"] = kmeds.medoid_indices_

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

    def compress(self):
        """
        Method to compress the distance matrix using affinity propagation
        Takes:
            distance_matrix: distance matrix
            damping: damping parameter for affinity propagation
            max_iter: maximum number of iterations for affinity propagation
            names: list of names of files in distance matrix
            return_inds_to_filter_boundary: boolean to add key to filter boundaries
        Returns:
            compressed_dictionary: dictionary with information about the clusters
        """
        compressed_dictionary = {}

        print(f"Estimated number of clusters: {self.cluster_results['n_clusters_']}")
        # get count of a value in a list
        temp_dict = dict(
            zip(
                sorted(list(self.cluster_results["best_performance"][0].keys())),
                self.cluster_results["cluster_centers_indices"],
            )
        )
        for k, v in temp_dict.items():
            compressed_dictionary[str(k)] = {
                "count": str(list(self.cluster_results.labels).count(k)),
                "index_center": str(v),
            }
            total_count = len(self.cluster_results.labels)
            if self.cluster_results.names != None:
                compressed_dictionary[str(i)]["name"] = self.cluster_results["names"][
                    self.cluster_results["cluster_centers_indices"][i]
                ]
        # compute percentage of each cluster
        for key in compressed_dictionary.keys():
            if type(key) == int:
                compressed_dictionary[key]["percentage"] = str(
                    float(compressed_dictionary[key]["count"])
                    / float(total_count)
                    * 100
                )
            else:
                if key.isnumeric():
                    compressed_dictionary[key]["percentage"] = str(
                        float(compressed_dictionary[key]["count"])
                        / float(total_count)
                        * 100
                    )

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

        # compute silhouette score
        """if len(set(labels)) == 1:
            silhouette_avg = 1.0
        else:
            silhouette_avg = silhouette_score(distance_matrix, labels)"""
        silhouette_avg = self.cluster_results["best_silhouette"]
        print(f"Silhouette Coefficient: {silhouette_avg}")
        # compressed_dictionary["boundary_inds"] = self.cluster_results["bounary_list_inds"]
        compressed_dictionary["silhouette"] = float(silhouette_avg)
        compressed_dictionary["labels"] = [
            int(i) for i in self.cluster_results["labels"]
        ]
        compressed_dictionary["n_clusters"] = int(self.cluster_results["n_clusters"])
        compressed_dictionary["total_count"] = int(total_count)
        return compressed_dictionary
