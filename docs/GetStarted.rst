Getting Started
===============

Examples for pycpet are located in the ``examples`` directory of the repository. The examples are designed to demonstrate how to use the package for various applications.

Running ``cpet.py``
-------------------

``cpet.py`` runs one calculation from an options file:

.. code-block:: console

    cpet.py -o options.json

Each run reads the options file, fills in defaults for anything left out, checks the options, and creates ``outputpath`` if needed. It then runs the method named in ``CPET_method``:

- Point, volume, topology and ``box_check`` methods run on every PDB or PQR file in ``inputpath`` (see :doc:`Input Files <InputFiles>`).
- Clustering, PCA and visualization methods read the outputs of earlier runs from ``inputpath``.

``volume``, ``volume_ESP``, ``topo`` and ``topo_GPU`` save one output file per structure, work through the structures in random order, and skip any that already have an output, so several copies of ``cpet.py`` can work through the same folder at once. ``point_field`` and ``point_mag`` write a single file with one line per structure.

Without ``-o``, ``cpet.py`` reads ``./options/options.json``. ``-i``, ``-p`` and ``-m`` override ``inputpath``, ``outputpath`` and ``CPET_method`` for one run, and ``-d`` prints debugging messages; see :ref:`command-line flags <command-line-flags>`.

The Options File
----------------

The options file is the cornerstone of *PyCPET*. It is a .json file that contains all relevant parameters you wish to specify for the calculation. Examples can be found in ``examples``. Every option, with its default, is listed in the :doc:`Options Reference <Options>`, and :doc:`Atom Specification <AtomSpec>` explains how to choose the ``center``, axes and filters.

Available Methods
-----------------

These are the tested methods for ``CPET_method``. Standalone methods can be run for single pdbs, or over any set of pdbs in a directory as long as they have a common orientable atom set:

- ``point_field``: Computing the electric field at a point, either arbitrary or defined by some atom selection.
- ``point_mag``: Computing the electric field magnitude at a point.
- ``volume``: Computing the electric field in a volume, defined by an oriented grid.
- ``volume_ESP``: Computing the electrostatic potential in a volume, defined by an oriented grid.
- ``topo``: Computing the electric field topology in a volume by distribution of streamlines (runs on several processes in parallel). See [Hennefarth2020]_ and [Dinh2008]_
- ``topo_GPU``: Same as above, but GPU-accelerated. See [Ajmera2025]_
- ``box_check``: Prints out any residues that are intruding in the box volume defined by the user

The below are methods that depend on one or more previous calculations run above. These provide immense value for dynamics analysis.

- ``cluster``: Clustering a set of electric field topologies to get representative fields
- ``cluster_volume``: Clustering a set of electric field volumes to get representative fields, using distances between the fields directly.
- ``cluster_volume_tensor``: Clustering a set of electric field volumes to get representative fields, using a tensor-decomposition. See [Chaturvedi2025]_
- ``cluster_volume_esp_tensor`` :bdg-secondary-line:`beta`: Clustering a set of electrostatic potential volumes to get representative electrostatic potentials, using a tensor-decomposition. This is the only method to use for electrostatic potential clustering.
- ``pca`` :bdg-secondary-line:`beta`: Performing PCA on a set of electric field volumes, especially useful for comparing field dynamics between mutants. See [Chaturvedi2024]_ and [Ajmera2025]_
- ``pca_compare`` :bdg-secondary-line:`beta`: Performing PCA separately on each of several sets of electric field volumes (e.g. variants), listed in ``inputpath_list``.
- ``visualize_field`` or ``visualize_esp``: Make a .bild file for ChimeraX to visualize the electric field or electrostatic potential, respectively.
