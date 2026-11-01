Dynamics: Clustering & PCA
==========================
Most of this section depends on prior computations of electric field, streamline distribution, or electrostatic potential. Please refer to :doc:`FieldTopESP` for details on how to compute these quantities.

Clustering groups the structures in ``inputpath``, usually frames of a simulation, by how similar their fields or potentials are. Each method builds a distance matrix between every pair of structures in its own way, then clusters it the same way.

Clusters and Output
-------------------

- Structures are clustered with K-Medoids (Partitioning Around Medoids) on the squared distances. Each cluster's medoid is a real structure, so it serves as the cluster's representative field.
- ``defined_n_clusters`` sets the number of clusters. Without it, 1 to 50 clusters are tried and the elbow of the inertia curve is chosen, which needs at least 50 structures.
- The log prints the number of clusters, the share of structures in each and the silhouette score (from −1 to 1; higher means better separated clusters).

``outputpath`` gets:

- ``<type>_file_list.txt``: the files that were clustered, in order.
- ``distance_matrix.dat.npy``: the distance matrix, in the same order.
- ``compressed_dictionary.json``: one entry per cluster, largest first, with its ``count`` and ``percentage``, its medoid (``name_center``), and its ``files`` sorted by distance from the medoid. It also has the overall ``silhouette``, ``n_clusters`` and ``total_count``.

With ``"cluster_reload": true``, the file list and distance matrix are read back from ``outputpath`` instead of being recomputed, so trying a different ``defined_n_clusters`` takes seconds. ``"plot_clusters": true`` shows the clusters in 3D (multidimensional scaling of the distance matrix), without saving the plots.

Electric Field Topologies
-------------------------

``cluster`` clusters the ``.top`` files from ``topo`` or ``topo_GPU`` [Hennefarth2020]_ [Ajmera2025]_.

- Each file becomes a 2D histogram of streamline distance against curvature, normalized to sum to 1.
- All histograms share the same bins, spanning the smallest to largest values over every file. Bin widths follow the Freedman–Diaconis rule, :math:`2\,\mathrm{IQR}/n^{1/3}`, with the interquartile range over all streamlines and :math:`n` the streamlines per file. The log prints both widths.
- The distance between two histograms :math:`f` and :math:`g` is the chi-squared distance, summed over bins :math:`i` and skipping bins that are empty in both:

.. math::

   d(f, g) = \frac{1}{2} \sum_{i} \frac{(f[i] - g[i])^2}{f[i] + g[i]}

Every file should have the same ``n_samples``. Otherwise *PyCPET* warns and uses the mean count for the bin widths.

The distance matrix is computed in Python, in parallel on every core. If it fails with ``can't start new thread`` (common on login nodes), limit the workers with ``LOKY_MAX_CPU_COUNT``, for example ``export LOKY_MAX_CPU_COUNT=4``.

Electric Field Tensor-Based Decomposition
-----------------------------------------

``cluster_volume_tensor`` clusters the ``_efield.dat`` files from ``volume`` using a tensor decomposition [Chaturvedi2025]_.

- The fields are stacked into one 5D tensor: structures × grid x × grid y × grid z × field component. All files must come from the same grid (``dimensions``, ``step_size``, ``x`` and ``y``).
- The tensor is approximated by a CP (PARAFAC) decomposition of rank :math:`R`, a sum of :math:`R` products of one vector per dimension. Each structure is then described by its :math:`R` weights along the structure dimension.
- The distance between two structures is the Euclidean distance between their weights.

The tensor is saved as ``5d_tensor.npy`` in ``outputpath``.

Choosing the rank
^^^^^^^^^^^^^^^^^

``rank`` sets it directly. Without it, *PyCPET* decomposes the tensor at every rank from 1 to ``max_rank`` (default 30) and logs each reconstruction error (mean squared error divided by the mean absolute value of the tensor):

- If the error at the elbow of that curve is at most ``tensor_threshold`` (default 0.001), the elbow rank is used.
- Otherwise, or if there is no elbow, the lowest rank with an error below ``tensor_threshold`` is used.
- If no rank reaches it, the rank with the lowest error is used and a warning is logged. Raise ``max_rank`` or ``tensor_threshold`` to change this.

Trying every rank is the slowest step and needs the whole tensor in memory. Once the log shows the errors, set ``rank`` for later runs.

``cluster_volume`` skips the decomposition and uses the field directly: the distance between two structures is the sum, over grid points, of the length of the difference between their field vectors.

Electrostatic Potential Tensor-Based Decomposition :bdg-secondary-line:`beta`
-----------------------------------------------------------------------------

``cluster_volume_esp_tensor`` clusters the ``_esp.dat`` files from ``volume_ESP`` the same way as fields, with one value per grid point instead of three. It is the only method for clustering potentials.

- The rank is chosen as for fields, from ``min_rank`` (default 1) to ``max_rank``.
- If ``outputpath`` already has a ``5d_tensor.npy`` made from these files, it is loaded instead of rebuilt, which saves time when rerunning. It is only reused if it has one entry per file and one value per grid point (so not a field tensor) and is newer than every input file; otherwise it is built again.
