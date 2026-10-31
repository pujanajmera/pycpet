Computing Fields, Topologies & Potentials
=========================================

Fields and potentials are sums over the charges left after filtering (see :doc:`Atom Specification <AtomSpec>`), computed in single precision by *PyCPET*'s C extension.

We recommend filtering out the residue that contains ``center``, so its own charges don't contribute to the field at its atoms. Example ``1A_point-field`` does this for residue CF1 378:

.. code-block:: json

    "filter_intersect": true,
    "filter": {
        "resid": ["CF1"],
        "resnum": [378]
    }

.. _point-fields:

Point Electric Fields
---------------------

``point_field`` gives the field vector at ``center``, and ``point_mag`` its magnitude, in V/Å:

.. math::

   \mathbf{E}(\mathbf{r}_0) = k \sum_{i=1}^{N} q_i \, \frac{\mathbf{r}_0 - \mathbf{r}_i}{\lvert \mathbf{r}_0 - \mathbf{r}_i \rvert^{3}}

where:

- :math:`\mathbf{r}_0` is ``center``
- :math:`\mathbf{r}_i` and :math:`q_i` are the position and partial charge (in e) of each of the :math:`N` charges
- :math:`k = 14.3996451` V·Å/e

There is no cutoff or softening, so a ``center`` exactly on an unfiltered atom (for example with ``"method": "first"``) gives NaN.

**Frame.** The field is in the input file's frame, or in the active-site frame if both ``x`` and ``y`` are given; ``x`` alone is an error. The magnitude is the same either way, as examples ``1A_point-field`` and ``1B_point-field-oriented`` show.

**Output.** ``outputpath`` gets one file, ``point_field.dat`` or ``point_mag.dat``, with a line per structure:

.. code-block:: text

    1_alcdehydro_run1:[-1.010131    0.5121924   0.28650615]

**Running.** Structures run one at a time on a single thread. Each run rewrites the output file, so several copies of ``cpet.py`` can't share a folder.

Volume Fields & Potentials
--------------------------

``volume`` gives the field (V/Å) and ``volume_ESP`` the potential (V) on a grid around ``center``. Both need ``x`` and ``y``, since the grid is built in the frame they define.

Grid
^^^^

- ``dimensions`` sets the box half widths and ``step_size`` the grid spacing, in Å. We recommend a ``step_size`` of 0.1 Å.
- Each axis gets 2 × ``dimensions`` / ``step_size`` intervals, rounded down, with points on both edges. Half widths of 0.5 Å give 11 × 11 × 11 = 1331 points (example ``2A_3D-field``).
- Floating point can lose a point: 2 × 0.15 / 0.1 evaluates to 2.999…, which gives 2 intervals, not 3.

Field and potential
^^^^^^^^^^^^^^^^^^^

The field uses the :ref:`point field <point-fields>` sum at each grid point, with :math:`\lvert \mathbf{r} - \mathbf{r}_i \rvert^2` kept above 10⁻⁶ Å² so an atom on a grid point doesn't give NaN.

The potential is

.. math::

   \phi(\mathbf{r}) = k \sum_{i=1}^{N} \frac{q_i}{\lvert \mathbf{r} - \mathbf{r}_i \rvert}

with the same :math:`k`, :math:`q_i` and :math:`N`. It has no such safeguard, and is saved at reduced precision (float16, about 3 significant digits).

Output and running
^^^^^^^^^^^^^^^^^^

- Each structure gets ``<name>_efield.dat`` or ``<name>_esp.dat`` in ``outputpath``.
- Each file starts with a header giving the number of points per axis, the half widths, the center and the basis matrix. Then each row is one grid point in the box frame: ``x y z Ex Ey Ez`` or ``x y z V``.
- Structures run in random order and finished ones are skipped, so several copies of ``cpet.py`` can share a folder.
- The field is computed for the whole grid in one call, on a single thread. The potential is computed one grid point at a time, with each sum parallelized over charges using OpenMP, so it follows ``OMP_NUM_THREADS``.

Electric Field Topologies
-------------------------

``topo`` and ``topo_GPU`` compute the distribution of field streamlines in a box around ``center`` [Hennefarth2020]_ [Dinh2008]_ [Ajmera2025]_. They need ``x``, ``y``, ``dimensions`` and ``step_size`` as for volumes, plus ``n_samples``, the number of streamlines.

.. important::

   For ``topo``, set ``OMP_NUM_THREADS=1`` and parallelize only with ``concur_slip`` in the options file. With more than one OpenMP thread, the CPU field calculation gives incorrect results. ``topo_GPU`` isn't affected.

Streamlines
^^^^^^^^^^^

- Start points fill the box on a regular grid (``initializer: "uniform"``, the default), with ``n_samples`` rounded up to the next perfect cube, or at random positions (``"random"``).
- Each streamline gets a random maximum number of steps, from 1 up to the box diagonal divided by ``step_size``. ``"max_streamline_init": "fixed_rand"`` fixes the random seed so runs are reproducible.
- Each step moves ``step_size`` Å along the field direction, using the :ref:`point field <point-fields>` sum. A streamline stops when it leaves the box or reaches its maximum steps.

For each streamline, *PyCPET* records the distance between its start and end, and the mean of its curvature at both ends, :math:`\kappa = \lvert \mathbf{r}' \times \mathbf{r}'' \rvert / \lvert \mathbf{r}' \rvert^{3}`, estimated from the next two steps.

Output and running
^^^^^^^^^^^^^^^^^^

- Each structure gets ``<name>.top`` in ``outputpath``, one row per streamline: distance, then curvature. :doc:`Dynamics <Dynamics>` turns these into histograms for clustering.
- Structures run in random order and finished ones are skipped, so several copies of ``cpet.py`` can share a folder.
- ``topo`` runs streamlines on ``concur_slip`` worker processes (default 4), in C. ``topo_GPU`` runs them on one GPU in batches of ``GPU_batch_freq`` steps; see :doc:`Parallel & GPU <Parallel>`.

Example ``3A_field-topology`` computes 5000 streamlines in a 1 Å box.
