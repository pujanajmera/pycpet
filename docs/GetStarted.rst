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

Your First Calculation
----------------------

``examples/1A_point-field`` computes the electric field at the midpoint of a C=O bond in residue 378 of an alcohol dehydrogenase variant, for two frames of a simulation. Its options file, ``options/options.json``, is:

.. code-block:: json

    {
        "CPET_method": "point_field",
        "inputpath": "./pdb",
        "center": {
            "method": "mean",
            "atoms": [
                {"C7": 378},
                {"O9": 378}
            ]
        },
        "filter_intersect": true,
        "filter": {
            "resid": ["CF1"],
            "resnum": [378]
        },
        "filter_radius": 40
    }

- ``"CPET_method": "point_field"`` computes the field vector at one point.
- ``"inputpath": "./pdb"`` points to the folder with the two frames, ``1_alcdehydro_run1.pdb`` and ``2_alcdehydro_run1.pdb``.
- ``center`` is the point itself: the mean position of atoms C7 and O9 of residue 378, which is the midpoint of the bond.
- There is no ``x`` or ``y``, so the field is given in the input file's frame rather than a frame aligned to the bond.
- ``filter`` with ``"filter_intersect": true`` removes residue CF1 378, so the residue's own charges don't contribute to the field at its bond.
- ``"filter_radius": 40`` keeps only residues with at least one atom within 40 Å of the center.
- ``outputpath`` isn't set, so results go to ``./outdir``.

Run it from the example's folder:

.. code-block:: console

    cd examples/1A_point-field
    cpet.py -o options/options.json

The log lists the atoms used for the center, the filters applied and each structure as it is processed. ``outdir/point_field.dat`` then has one line per structure, with the x, y and z components of the field:

.. code-block:: text

    1_alcdehydro_run1:[-1.010131    0.5121924   0.28650615]
    2_alcdehydro_run1:[-1.0865108   0.33583832  0.3645109 ]

Choosing a Method
-----------------

These are the tested methods for ``CPET_method``. Standalone methods can be run for single pdbs, or over any set of pdbs in a directory as long as they have a common orientable atom set:

.. list-table::
   :header-rows: 1
   :widths: 28 57 15

   * - Method
     - Description
     - Guide
   * - ``point_field``
     - Computing the electric field at a point, either arbitrary or defined by some atom selection.
     - :doc:`Fields <FieldTopESP>`
   * - ``point_mag``
     - Computing the electric field magnitude at a point.
     - :doc:`Fields <FieldTopESP>`
   * - ``volume``
     - Computing the electric field in a volume, defined by an oriented grid.
     - :doc:`Fields <FieldTopESP>`
   * - ``volume_ESP``
     - Computing the electrostatic potential in a volume, defined by an oriented grid.
     - :doc:`Fields <FieldTopESP>`
   * - ``topo``
     - Computing the electric field topology in a volume by distribution of streamlines (runs on several processes in parallel). See [Hennefarth2020]_, [Dinh2008]_ and [Ajmera2025]_
     - :doc:`Fields <FieldTopESP>`
   * - ``topo_GPU``
     - Same as above, but GPU-accelerated. See [Ajmera2025]_
     - :doc:`Parallel & GPU <Parallel>`
   * - ``box_check``
     - Prints out any residues that are intruding in the box volume defined by the user
     - :doc:`Fields <FieldTopESP>`

The below are methods that depend on one or more previous calculations run above. These provide immense value for dynamics analysis.

.. list-table::
   :header-rows: 1
   :widths: 28 57 15

   * - Method
     - Description
     - Guide
   * - ``cluster``
     - Clustering a set of electric field topologies to get representative fields
     - :doc:`Dynamics <Dynamics>`
   * - ``cluster_volume``
     - Clustering a set of electric field volumes to get representative fields, using distances between the fields directly.
     - :doc:`Dynamics <Dynamics>`
   * - ``cluster_volume_tensor``
     - Clustering a set of electric field volumes to get representative fields, using a tensor-decomposition. See [Chaturvedi2025]_
     - :doc:`Dynamics <Dynamics>`
   * - ``cluster_volume_esp_tensor`` :bdg-secondary-line:`beta`
     - Clustering a set of electrostatic potential volumes to get representative electrostatic potentials, using a tensor-decomposition. This is the only method to use for electrostatic potential clustering.
     - :doc:`Dynamics <Dynamics>`
   * - ``pca`` :bdg-secondary-line:`beta`
     - Performing PCA on a set of electric field volumes, especially useful for comparing field dynamics between mutants. See [Chaturvedi2024]_ and [Ajmera2025]_
     - :doc:`Dynamics <Dynamics>`
   * - ``pca_compare`` :bdg-secondary-line:`beta`
     - Performing PCA separately on each of several sets of electric field volumes (e.g. variants), listed in ``inputpath_list``.
     - :doc:`Dynamics <Dynamics>`
   * - ``visualize_field``, ``visualize_esp``
     - Make a .bild file for ChimeraX to visualize the electric field or electrostatic potential, respectively.
     - :doc:`Visualization <Visualization>`

Next Steps
----------

Each of these examples runs as is from its folder with ``cpet.py -o options/options.json``. All use the same two frames as the first calculation.

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - Example
     - What it adds
     - Guide
   * - ``1B_point-field-oriented``
     - Orients the field along the C=O bond with ``x`` and ``y``; compare with ``1A`` to see the magnitude is unchanged.
     - :doc:`Atom Specification <AtomSpec>`
   * - ``2A_3D-field``
     - Computes the field on a 3D grid around the bond, with ``dimensions`` and ``step_size``, writing one ``_efield.dat`` file per frame.
     - :doc:`Fields <FieldTopESP>`
   * - ``2B_3D-field-visualization``
     - Turns field files into ``.bild`` files for ChimeraX, using the ``visualization`` options.
     - :doc:`Visualization <Visualization>`
   * - ``3A_field-topology``
     - Computes the streamline distribution with ``topo``, writing one ``.top`` file per frame of streamline distances and curvatures.
     - :doc:`Fields <FieldTopESP>`
