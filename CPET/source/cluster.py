import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns

from sklearn.cluster import AffinityPropagation, HDBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
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
        if not options["cluster_method"]:
            print("No cluster method specified, defaulting to K-Medoids")
            self.cluster_method = "kmeds"
        else:
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
        self.cluster_analyze()

    def kmeds(self):
        cluster_results = {}
        distance_matrix = self.distance_matrix
        distance_matrix = distance_matrix**2
        silhouette_list = []
        for i in range(10):
            kmeds = KMedoids(
                n_clusters = i + 2, 
                random_state = 0, 
                metric = "precomputed", 
                method = "pam",
                init = "k-medoids++"
            )
            kmeds.fit(distance_matrix)
            labels = list(kmeds.labels_)
            score = silhouette_score(distance_matrix, 
                                     labels, 
                                     metric = "precomputed"
            )
            print(i + 2, score)
            silhouette_list.append(score)
        max_index = silhouette_list.index(max(silhouette_list))
        print(
            f"Using {max_index+2} number of clusters with Partitioning around Medoids (PAM)"
        )
        kmeds = KMedoids(
            n_clusters = max_index + 2, 
            random_state = 0, 
            metric = "precomputed", 
            method = "pam", 
            init = "k-medoids++"
        )
        kmeds.fit(distance_matrix)
        cluster_results["labels"] = list(kmeds.labels_)
        cluster_results["silhouette_score"] = silhouette_score(distance_matrix, 
                                                               cluster_results["labels"], 
                                                               metric = "precomputed")
        cluster_results["n_clusters"] = max_index + 2
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
        compressed_dictionary = {}
        # get count of a value in a list
        for i in range(self.n_clusters):
            temp_dict = {}
            temp_dict["count"] = list(self.cluster_results["labels"]).count(i)
            temp_dict["index_center"] = self.cluster_results.cluster_centers_indices[i]
            temp_dict["name_center"] = self.topo_file_list[temp_dict["index_center"]]
            temp_dict["percentage"] = float(temp_dict["count"]) / float(
                len(self.cluster_results["labels"])
            ) * 100
            cluster_indices = [y for y, x in enumerate(self.cluster_results["labels"]) if x == i]
            temp_dict["mean_distance"] = np.mean(self.distance_matrix[temp_dict["index_center"]][cluster_indices])
            temp_dict["max_distance"] = np.max(self.distance_matrix[temp_dict["index_center"]][cluster_indices])
            temp_zip = zip(
                            self.topo_file_list[cluster_indices], 
                            list(self.distance_matrix[temp_dict["index_center"]][cluster_indices])
                       )
            sorted_temp_zip = sorted(temp_zip, key = lambda x: x[1])
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
            #Plot clusters with Multi-Dimensional Scaling
            mds = MDS(n_components=3, dissimilarity="precomputed", random_state = 0)
            projection = mds.fit_transform(self.distance_matrix)  # Directly feed the distance matrix
            color_palette = sns.color_palette('deep', 12)
            cluster_colors = [color_palette[label] for label in self.cluster_results.labels_]

            # Define different perspectives
            perspectives = [
                (30, 30),   # Elevation=30°, Azimuth=30°
                (30, 120),  # Elevation=30°, Azimuth=120°
                (30, 210),  # Elevation=30°, Azimuth=210°
                (30, 300),  # Elevation=30°, Azimuth=300°
                (90, 0)     # Top view
            ]

            # Create a 3D scatter plot from different perspectives
            for elev, azim in perspectives:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], s=10, linewidth=0, alpha=0.25, c=cluster_colors)
                ax.view_init(elev, azim)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
                plt.title(f'View from elevation {elev}°, azimuth {azim}°')
                plt.show()
        if self.plot_dwell_times == True:
            print(0)
        return compressed_dictionary
