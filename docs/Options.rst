Options Reference
=================

Every key that ``cpet.py`` reads from the options file, grouped by what it controls. Keys left out of the options file take the default shown. Lengths are in Angstroms.

See :doc:`Getting Started <GetStarted>` for a walkthrough of the most common options, and :doc:`Atom Specification <AtomSpec>` for the syntax of ``center``, ``x``, ``y`` and ``filter``.

General
-------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``CPET_method``
     - all
     - required
     - The calculation to run. One of ``point_field``, ``point_mag``, ``volume``, ``volume_ESP``, ``topo``, ``topo_GPU``, ``box_check``, ``cluster``, ``cluster_volume``, ``cluster_volume_tensor``, ``cluster_volume_esp_tensor``, ``visualize_field``, ``visualize_esp``, ``pca``, ``pca_compare``.
   * - ``inputpath``
     - all
     - ``"./inpdir"``
     - Folder of input files: PDB/PQR structures for field calculations, or the output of an earlier calculation for clustering, PCA and visualization.
   * - ``outputpath``
     - all
     - ``"./outdir"``
     - Folder for results. Created if it doesn't exist.
   * - ``inputfiletype``
     - field calculations
     - ``"pdb"``
     - ``"pdb"`` or ``"pqr"``. See :doc:`Input Files <InputFiles>` for the format each needs.

Orientation and box
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``center``
     - field calculations
     - required
     - Origin of the calculation; for point methods, where the field is computed. A list ``[x, y, z]`` or an atom selection.
   * - ``x``
     - field calculations
     - none
     - Point that defines the x-axis direction from ``center``. Without ``x``, the input file's frame is used and ``y`` is ignored.
   * - ``y``
     - field calculations
     - none
     - Point used to fix the y-axis. ``volume``, ``volume_ESP``, ``topo`` and ``topo_GPU`` only build their grid when both ``x`` and ``y`` are given.
   * - ``dimensions``
     - ``volume``, ``volume_ESP``, ``topo``, ``topo_GPU``, ``filter_in_box``
     - none
     - Box **half** widths ``[a, b, c]``; ``[0.5, 0.5, 0.75]`` is a 1 x 1 x 1.5 box.
   * - ``step_size``
     - ``volume``, ``volume_ESP``, ``topo``, ``topo_GPU``
     - none
     - Grid spacing for volumes; streamline step length for topologies (a convergence parameter).
   * - ``box_shift``
     - field calculations
     - ``[0, 0, 0]``
     - Moves the box away from ``center`` by this vector, in the oriented frame. The code shifts the protein the opposite way.
   * - ``dtype``
     - field calculations
     - ``"float32"``
     - Floating point precision of the grid and GPU arrays: ``"float32"`` or ``"float64"``.

Filtering
---------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``filter``
     - field calculations
     - none
     - Removes the matching atoms from the charges used. Keys: ``atom_number``, ``atom_type``, ``resid``, ``resnum``, ``chain``.
   * - ``filter_intersect``
     - ``filter``
     - ``false``
     - ``false`` removes atoms matching any key; ``true`` matches the lists position by position.
   * - ``filter_radius``
     - field calculations
     - none
     - Keeps only residues with at least one atom within this distance of ``center``; whole residues outside it are removed. Not allowed with ``box_check``.
   * - ``filter_in_box``
     - field calculations
     - ``false``
     - Removes charges inside the box set by ``dimensions``. Always off for ``box_check``.

Topology
--------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``n_samples``
     - ``topo``, ``topo_GPU``
     - none (required)
     - Number of streamlines, a convergence parameter. With the ``uniform`` initializer it is rounded up to the next perfect cube.
   * - ``initializer``
     - ``topo``, ``topo_GPU``
     - ``"uniform"``
     - Where streamlines start: ``"uniform"`` (a regular grid) or ``"random"``.
   * - ``max_streamline_init``
     - ``topo``, ``topo_GPU``
     - ``"true_rand"``
     - With the ``uniform`` initializer, ``"fixed_rand"`` fixes the random seed for streamline lengths so runs are reproducible.
   * - ``concur_slip``
     - ``topo``
     - ``4``
     - Number of worker processes. See :doc:`Parallel & GPU <Parallel>`.
   * - ``GPU_batch_freq``
     - ``topo_GPU``
     - ``100``
     - Streamline steps computed per GPU batch.

Clustering
----------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``cluster_method``
     - clustering methods
     - ``"kmeds"``
     - ``"kmeds"`` (K-Medoids). ``"affinity"`` (affinity propagation) and ``"hdbscan"`` are also accepted :bdg-warning-line:`not working yet`. Unknown values fall back to ``"kmeds"``.
   * - ``defined_n_clusters``
     - ``kmeds``
     - none
     - Fixed number of clusters (integer). If unset, 1–50 clusters are tried and the elbow is chosen, which needs more than 50 structures.
   * - ``cluster_reload``
     - clustering methods
     - ``false``
     - Reuse the file list and distance matrix saved in ``outputpath`` instead of recomputing them.
   * - ``plot_clusters``
     - clustering methods
     - ``false``
     - Show 3D plots of the clusters (multidimensional scaling of the distance matrix) in a plot window. The plots are not saved.
   * - ``plot_dwell_times``
     - clustering methods
     - ``false``
     - :bdg-warning-line:`not working yet` Setting it only logs a warning.
   * - ``rank``
     - tensor clustering
     - none
     - Tensor decomposition rank (integer). If unset, it is chosen from ``tensor_threshold``.
   * - ``tensor_threshold``
     - tensor clustering
     - ``0.001``
     - Error threshold for choosing the rank automatically.
   * - ``max_rank``
     - tensor clustering
     - ``30``
     - Highest rank tried when choosing automatically.
   * - ``min_rank``
     - ``cluster_volume_esp_tensor``
     - ``1``
     - Lowest rank tried when choosing automatically.

PCA
---

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``n_pca_components``
     - ``pca``, ``pca_compare``
     - ``10``
     - Number of principal components.
   * - ``field_file_list``
     - ``pca``, ``pca_compare``
     - all ``.dat`` files in ``inputpath``
     - Explicit list of field files to use.
   * - ``save_pca``
     - ``pca``, ``pca_compare``
     - ``false``
     - Save the fitted PCA to ``outputpath/pca.pkl``.
   * - ``pca_reload``
     - ``pca``, ``pca_compare``
     - ``false``
     - Load the fitted PCA from ``outputpath/pca.pkl`` instead of fitting.
   * - ``whitening``
     - ``pca``, ``pca_compare``
     - ``false``
     - Whiten the principal components.
   * - ``verbose``
     - ``pca``, ``pca_compare``
     - ``true``
     - Print extra PCA output.
   * - ``inputpath_list``
     - ``pca_compare``
     - required
     - List of folders, one per variant.
   * - ``outputpath_list``
     - ``pca_compare``
     - ``<input>/pca_out`` for each
     - List of output folders, one per entry in ``inputpath_list``.
   * - ``pca_combined_only``
     - ``pca_compare``
     - ``false``
     - :bdg-warning-line:`not working yet` Combined PCA across variants: ``true`` skips the per-variant PCA and currently produces no PCA output.

Visualization
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``visualization``
     - ``visualize_field``, ``visualize_esp``
     - ``{"cutoff": 0, "sparsify_factor": 1}``
     - ``cutoff`` is a percentile (0–100) of magnitudes below which nothing is drawn. ``sparsify_factor`` thins the grid for ``visualize_field`` only, so larger values draw fewer arrows. Keys you leave out keep their defaults.
   * - ``max_p_esp``, ``min_p_esp``
     - ``visualize_esp``
     - from the data
     - Color scale limits for positive potentials.
   * - ``max_n_esp``, ``min_n_esp``
     - ``visualize_esp``
     - from the data
     - Color scale limits for negative potentials.

Output and debugging
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Option
     - Used by
     - Default
     - Description
   * - ``write_transformed_pdb``
     - field calculations
     - ``false``
     - Write ``transform_<name>.pdb`` (oriented and shifted coordinates) to the current working directory, for making figures. See :doc:`Visualization <Visualization>`.
   * - ``strip_filter``
     - ``write_transformed_pdb``
     - ``false``
     - Leave filtered atoms out of the transformed PDB.
   * - ``profile``
     - field calculations
     - ``false``
     - Developer option for profiling.

.. _command-line-flags:

Command-line flags
------------------

``cpet.py`` takes these flags; ``-i``, ``-p`` and ``-m`` override the options file.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Flag
     - Effect
   * - ``-o PATH``
     - Options file to read (default ``./options/options.json``).
   * - ``-i PATH``
     - Sets ``inputpath``.
   * - ``-p PATH``
     - Sets ``outputpath``.
   * - ``-m METHOD``
     - Sets ``CPET_method``.
   * - ``-d``, ``--debug``
     - Print debugging messages.
