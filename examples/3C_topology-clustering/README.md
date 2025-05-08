# Oriented point electric fields

This example contains a .json options file to cluster topologies of the 3-D electric field surrounding the midpoint of a C=O bond in a CXF residue, in a mutated alcohol dehydrogenase. Since any legitimate clustering of molecular dynamics would require several hundred files, this is not a realistic example.

Briefly, there are a few output files. First is topo_file_list.txt, which is a list of all your topologies. Next is distance_matrix.dat.npy, which is your chi-squared distance matrix. Lastly is your compressed_dictionary.json, which has all information related to the clustering results. Although this file is massive, it is information-rich and is easily parseable for further analysis (e.g. dynamics, visualization of cluster centers, etc).