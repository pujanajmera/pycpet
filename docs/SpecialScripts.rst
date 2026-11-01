Special Scripts
=================

The following are tutorials for special scripts that aren't built into the cpet.py executable, but use the pycpet libraries to do some neat analyses.

.. note::

   Only ``cpet.py`` is installed as a command. These scripts are in ``CPET/source/scripts`` in the repository; run them with ``python`` and their path.

Electrostatic Interaction Energy Analysis
-------------------------------------------------
Authors: Anubhav Goswami, Pujan Ajmera. Credit to the Binju Wang group for the original method (w/o nuclear correction): [Yan2023]_

The script is electrostatic_interaction_QM.py, and it computes interaction energies (for QM/MM calculations/QM calculations with point charge embedding) using these equations:

.. math::
    E_{int,elec} = \sum_{i=1}^{N} \int_{V} \frac{q_i \rho(\mathbf{r})}{|\mathbf{r} - \mathbf{r}_i|} d\mathbf{r}
.. math::
    E_{int,nuc} = \sum_{i=1}^{N} \sum_{j=1}^{M} \frac{q_i Q_j}{r_{ij}}
.. math::
    E_{int} = E_{int,elec} + E_{int,nuc}

where :math:`q_i` and :math:`Q_j` are the partial charges of MM region of interest and the nuclear charges of the QM region, respectively. These are the following options that can be passed to the script:

- `-m/--molden`: The molden input file, which contains the QM density in a molden format.
- `-p/--pdb`: The path to the a PDB file that contains charges of all MM atoms.
- `-r/--res`: Flag for residue breakdown. If provided, analysis will be done by residue as well, and saved to residue_breakdown.txt in the working directory.
- `-o/--options`: PyCPET options file. Please see the :doc:`Get Started <../GetStarted>` page for more information on how to write an options file. This is solely used in this script to filter out the QM region atoms, otherwise the interaction energy will be unrealistic.
- `-v/--verbose`: Flag for verbose output. If provided, the script will tell you what interaction it is computing.
- `-s/--spin`: Total spin S of the QM region (default 0.5). Use 0 for closed-shell systems, otherwise the molecule won't build.
- `-c/--charge`: Total charge of the QM region, as an integer (default 0).

These interaction energies are output to the command line, so the following usage is recommended, from the repository folder:

.. code-block:: console

    python CPET/source/scripts/electrostatic_interaction_QM.py -m name_of_molden_file -p protein.pdb -o options.json -r > interaction_energies.txt

The Alexandrova group's work on Chorismate Mutase inspired this script [Chaturvedi2025]_

Two-Site 3D Electrostatic Potential Clustering :bdg-secondary-line:`beta`
-------------------------------------------------------------------------
Author: Pujan Ajmera

The script is tensor_based_cluster_double.py, and it provides representative clusters from a set of two 3D electrostatic potentials, for example at two active sites of the same structures. The clusters are "co-clusters", i.e. depend on both potentials changing simultaneously.

Each site's potentials are decomposed separately, as in :doc:`ESP tensor clustering <Dynamics>`, giving every structure a set of weights per site. The two sets are joined side by side, and structures are clustered by the Euclidean distance between the joined weights, using K-Medoids. The weights aren't rescaled, so a site with larger potentials counts more towards the distance.

These are the following options that can be passed to the script:

- `-d1/--dir1`: Folder with the ``_esp.dat`` files of the first site, from ``volume_ESP``.
- `-d2/--dir2`: Folder with the ``_esp.dat`` files of the second site. Both folders must contain exactly the same file names, so each structure is matched between them.
- `-u/--outputpath`: Folder for the results, created if needed.
- `-o`: PyCPET options file (default ./options/options.json). Only ``defined_n_clusters``, ``rank``, ``tensor_threshold`` and ``max_rank`` are read, as described in :doc:`Dynamics <Dynamics>`. If ``rank`` isn't set, it is chosen separately for each site.

The output folder gets the file lists (esp_file_list1.txt, esp_file_list2.txt and combined_file_list.txt), each site's weights (factor_mat1.npy, factor_mat2.npy), the joined weights (combined_factor_mat.npy), the distance matrix (distance_matrix_combined.npy) and compressed_dictionary.json, in the same format as for ``cpet.py`` clustering. The script isn't restartable, so every run starts from the beginning. The following usage is recommended, from the repository folder:

.. code-block:: console

    python CPET/source/scripts/tensor_based_cluster_double.py -d1 site1_esp -d2 site2_esp -u cluster_out -o options.json > cluster_log.txt

Field Topology from Dipole Moments :bdg-secondary-line:`beta`
-------------------------------------------------------------
Authors: Pujan Ajmera, Anubhav Goswami

The script is field_topology_dipole.py, and it computes the distribution of streamlines (see :doc:`Fields <FieldTopESP>`) from point dipoles instead of point charges, using the dipole field:

.. math::
    \mathbf{E}(\mathbf{r}) = k \sum_{i=1}^{N} \frac{3 (\boldsymbol{\mu}_i \cdot \mathbf{R}_i) \mathbf{R}_i - \boldsymbol{\mu}_i \lvert \mathbf{R}_i \rvert^2}{\lvert \mathbf{R}_i \rvert^5}, \quad \mathbf{R}_i = \mathbf{r} - \mathbf{r}_i

where :math:`\mathbf{r}_i` and :math:`\boldsymbol{\mu}_i` are the position and dipole moment of each of the :math:`N` dipoles.

Each dipole file (``.dip``) has a header line, then one line per dipole with 7 columns: index, x, y and z (Å), and the dipole moment's x, y and z components (a.u.). These are the following options that can be passed to the script:

- `-d/--dipole_dir`: Folder of ``.dip`` files. Every file is processed, in random order.
- `-c/--center`: The box center, as 3 coordinates (default 0 0 0).
- `-x`: A point the x axis points to from the center, as 3 coordinates (default 1 0 0).
- `-y`: A point that sets the y axis, as for ``y`` in the options file (default 0 1 0).
- `-dim/--dimensions`: The box half widths in Å (default 1 1 1).
- `-n/--n-samples`: The number of streamlines (default 1000), rounded up to the next perfect cube.
- `-s/--step-size`: The streamline step size in Å (default 0.1).
- `-t/--threads`: The number of worker processes for streamlines (default 1). As for ``topo``, set ``OMP_NUM_THREADS=1``.

Each dipole file gets ``<name>.top`` in the working directory, in the same format as from ``topo``, so it can be clustered with ``cluster``. The maximum number of steps per streamline is random and not seeded, so runs aren't reproducible. The following usage is recommended, from the repository folder:

.. code-block:: console

    OMP_NUM_THREADS=1 python CPET/source/scripts/field_topology_dipole.py -d dipoles -c 0 0 0 -x 1 0 0 -y 0 1 0 -dim 1 1 1 -n 1000 -s 0.1 -t 4

Residue Breakdown Analysis of Field Topology :bdg-secondary-line:`beta`
-----------------------------------------------------------------------
Authors: Pujan Ajmera

The script is residue_breakdown_analysis.py, and it ranks how much each residue shapes the field topology. For each structure and each residue, it recomputes the topology from that residue's charges alone, and compares it with the full topology using the chi-squared distance between histograms (see :doc:`Dynamics <Dynamics>`). A low distance means the residue on its own reproduces much of the full topology.

These are the following options that can be passed to the script:

- `-t/--tops`: Folder of ``.top`` files from ``topo``. Other files in the folder are ignored.
- `-p/--pdbs`: Folder of the matching PDB files. Every PDB file needs a ``.top`` file of the same name, and every ``.top`` file a PDB file. Other files in the folder are ignored.
- `-c/--curvbound`: Lower and upper curvature bounds of the histograms, separated by a comma (e.g. 0,2).
- `-cx/--curvres`: Curvature bin width of the histograms.
- `-d/--distbound`: Lower and upper distance bounds of the histograms, separated by a comma.
- `-dx/--distres`: Distance bin width of the histograms.
- `-o`: PyCPET options file (default ./options/options.json). Use the one from the ``topo`` run, with ``"max_streamline_init": "fixed_rand"``, so each residue's streamlines start from the same points and run for the same number of steps as the full topology. Residues are taken from the first PDB file after filtering, skipping water (WAT); a structure without one of them is left out of that residue's average.

Each residue's distance is printed for every structure as it is computed. The working directory gets residue_list.txt, the residues analyzed, and chi_squared_avg.txt, each residue's mean and standard deviation of the distance over all structures, lowest first. One topology is computed per residue per structure, so this is slow for large systems; as for ``topo``, set ``OMP_NUM_THREADS=1``. The following usage is recommended, from the repository folder:

.. code-block:: console

    OMP_NUM_THREADS=1 python CPET/source/scripts/residue_breakdown_analysis.py -t tops -p pdbs -c 0,2 -cx 0.02 -d 0,2 -dx 0.02 -o options.json
