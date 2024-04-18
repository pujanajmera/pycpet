import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx
from sklearn_extra.cluster import KMedoids
from glob import glob

class cluster:
    def __init__(self, options):
        self.cluster_method = options["cluster_method"]
        self.inputpath = self.options["inputpath"]
        self.outputpath = self.options["outputpath"]

    def Cluster(self):
        if self.cluster_method == "affinity":
            self.cluster_results = self.affinity()
        elif self.cluster_method == "kmeds":
            self.cluster_results = self.kmeds()
        elif self.cluster_method == "hdbscan":
            self.cluster_results = self.hdbscan()
        
        self.compress()

    def kmeds(self):
        distance_matrix = self.distance_matrix
        silhouette_list=[]
        for i in range(6):
            kmeds = KMedoids(n_clusters=i+2, random_state=0, metric="precomputed", method="pam")
            kmeds.fit(distance_matrix)
            labels = list(kmeds.labels_)
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            print(i+2, score)
            silhouette_list.append(score)
        max_index = silhouette_list.index(max(silhouette_list))
        print(f"Using {max_index+2} number of clusters with Partitioning around Medoids (PAM)")
        kmeds = KMedoids(n_clusters=max_index+2, random_state=0, metric="precomputed", method="pam")
        kmeds.fit(distance_matrix)
        labels = list(kmeds.labels_)
        n_clusters_ = max_index+2
        cluster_centers_indices = kmeds.medoid_indices_

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

        
        silhouette_list=[]
        for i in range(6):
            kmeds = KMedoids(n_clusters=i+2, random_state=0, metric="precomputed", method="pam")
            kmeds.fit(distance_matrix)
            labels = list(kmeds.labels_)
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            print(i+2, score)
            silhouette_list.append(score)
        max_index = silhouette_list.index(max(silhouette_list))
        print(f"Using {max_index+2} number of clusters with Partitioning around Medoids (PAM)")
        kmeds = KMedoids(n_clusters=max_index+2, random_state=0, metric="precomputed", method="pam")
        kmeds.fit(distance_matrix)
        labels = list(kmeds.labels_)
        n_clusters_ = max_index+2
        cluster_centers_indices = kmeds.medoid_indices_

        if return_inds_to_filter_boundary:
            print("Filtering...")
            # construct networkx graph
            bounary_list_inds = []

            # compute the 0.1 quantile of the distance matrix
            cutoff_distance = np.quantile(distance_matrix, filtered_cutoff)

            G = nx.Graph()
            G.add_nodes_from(range(len(labels)))
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if distance_matrix[i, j] < cutoff_distance:
                        G.add_edge(i, j, weight=distance_matrix[i, j])
            # add the labels to the graph
            for i in range(len(labels)):
                G.nodes[i]["label"] = labels[i]

            # iterate through the nodes and neighbors
            for i in range(len(labels)):
                # if i<20:
                neighbor_nodes = list(G.neighbors(i))
                neighbor_labels = [G.nodes[j]["label"] for j in neighbor_nodes]
                neighbor_setlist = list(set(neighbor_labels))
                self_label = G.nodes[i]["label"]
                if len(neighbor_setlist) > 0:
                    if len(neighbor_setlist) > 1:
                        bounary_list_inds.append(i)
                    else:
                        if self_label != neighbor_setlist:
                            bounary_list_inds.append(i)
        else:
            bounary_list_inds = []
        #Make graph of clusters:
        '''print("Making graph...")
        G = nx.Graph()
        for i in range(len(distance_matrix)):
            for j in range(i + 1, len(distance_matrix)):
                if distance_matrix[i][j] != 0:  # Assuming 0 indicates no edge
                    # Inverting the distance to use as weight because lower distance means stronger connection
                    G.add_edge(i, j, weight=1.0/distance_matrix[i][j])

        # Generate a color map that is dependent on the number of unique labels
        unique_labels = set(labels)
        n_colors = len(unique_labels)
        color_palette = plt.cm.viridis(np.linspace(0, 1, n_colors))  # Using a colormap to generate distinct colors
        color_map = {label: color for label, color in zip(unique_labels, color_palette)}

        # Assign colors to nodes based on labels
        node_colors = [color_map[label] for label in labels]

        # Use a spring layout to visualize the graph
        pos = nx.spring_layout(G, weight='weight')

        # Draw the graph
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, connectionstyle='arc3,rad=0.1')
        plt.title("Graph Visualization of Distance Matrix")
        plt.savefig("cluster_plot.png")'''
        print(f"Estimated number of clusters: {n_clusters_}")
        # get count of a value in a list
        temp_dict = dict(zip(sorted(list(best_performance[0].keys())),cluster_centers_indices))
        for k,v in temp_dict.items():
            compressed_dictionary[str(k)] = {
                "count": str(list(labels).count(k)),
                "index_center": str(v),
            }
            total_count = len(labels)
            if names != None:
                compressed_dictionary[str(i)]["name"] = names[cluster_centers_indices[i]]
        # compute percentage of each cluster
        for key in compressed_dictionary.keys():
            if type(key) == int:
                compressed_dictionary[key]["percentage"] = str(
                    float(compressed_dictionary[key]["count"]) / float(total_count) * 100
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
        '''if len(set(labels)) == 1:
            silhouette_avg = 1.0
        else:
            silhouette_avg = silhouette_score(distance_matrix, labels)'''
        silhouette_avg = best_silhouette
        print(f"Silhouette Coefficient: {silhouette_avg}")
        compressed_dictionary["boundary_inds"] = bounary_list_inds
        compressed_dictionary["silhouette"] = float(silhouette_avg)
        compressed_dictionary["labels"] = [int(i) for i in labels]
        compressed_dictionary["n_clusters"] = int(n_clusters_)
        compressed_dictionary["total_count"] = int(total_count)
        return compressed_dictionary
        
    def affinity(self):
        affinity = AffinityPropagation(
            affinity="precomputed", damping=damping, max_iter=max_iter
        )
        affinity_matrix=distance_matrix
        affinity_matrix[affinity_matrix < 0.2] = 0
        affinity.fit(affinity_matrix)
        cluster_centers_indices = affinity.cluster_centers_indices_
        labels = list(affinity.labels_)
        n_clusters_ = len(cluster_centers_indices)

    def hdbscan(self):
        performance_list = []
        for percentile_threshold in [70,80,90,99,99.9,99.99,99.999,99.9999,100]:
            threshold=np.percentile(distance_matrix.flatten(),percentile_threshold)
            filtered_distance_matrix=distance_matrix.copy()
            filtered_distance_matrix[filtered_distance_matrix > threshold] = 1
            clustering = HDBSCAN(min_samples=1,min_cluster_size=50,store_centers='medoid',copy=True, allow_single_cluster=False, n_jobs=-1)
            clustering.fit(filtered_distance_matrix)
            labels = clustering.labels_
            count_dict = {}
            for i in labels:
                if i in count_dict:
                    count_dict[i] += 1
                else:
                    count_dict[i] = 1
            score = silhouette_score(filtered_distance_matrix,labels)
            performance_list.append([count_dict, score, clustering, threshold, percentile_threshold])
            print(percentile_threshold, score)
            #filter out too noisy convergences
        performance_list_filtered = [i for i in performance_list if (i[0].get(-1, float('inf')) <= 250)]
        silhouettes = [i[1] for i in performance_list_filtered]
        if not silhouettes == []:
            best_silhouette = max(silhouettes)
        else:
            print("Significant noisy data (>5%) found here; take these results with a grain of salt")
            performance_list_filtered = performance_list
            silhouettes = [i[1] for i in performance_list_filtered]
            best_silhouette = max(silhouettes)
        best_performance = performance_list_filtered[len(silhouettes) - 1 - silhouettes[::-1].index(best_silhouette)]
        labels = best_performance[2].labels_
        best_filtered_matrix = distance_matrix.copy()
        best_filtered_matrix[best_filtered_matrix > best_performance[3]] = 1
        cluster_centers_indices = [np.where(np.all(best_filtered_matrix == best_performance[2].medoids_[i], 
                                                                    axis=1))[0][0] for i in range(len(best_performance[2].medoids_))]
        #Define number of clusters to include the 'noise' cluster
        n_clusters_ = len(best_performance[0].keys())
        #If noise cluster exists, add a null cluster center index
        if not n_clusters_ == len(best_performance[2].medoids_):
            cluster_centers_indices = ['']+cluster_centers_indices
        
        print(f"Best performance with {best_performance[4]}% cutoff and silhouette score of {best_silhouette}")

