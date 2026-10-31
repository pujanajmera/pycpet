Computing Fields, Topologies & Potentials
=========================================

Point Electric Fields
---------------------

``point_field`` computes the electric field vector at ``center``, and ``point_mag`` computes its magnitude, in V/Å. Each runs on every structure in ``inputpath``, using the charges left after any filters (see :doc:`Atom Specification <AtomSpec>`).

The field is a direct Coulomb sum over those charges:

.. math::

   \mathbf{E}(\mathbf{r}_0) = k \sum_{i=1}^{N} q_i \, \frac{\mathbf{r}_0 - \mathbf{r}_i}{\lvert \mathbf{r}_0 - \mathbf{r}_i \rvert^{3}}

where :math:`\mathbf{r}_0` is ``center``, :math:`\mathbf{r}_i` and :math:`q_i` are the position and partial charge (in units of e) of each of the :math:`N` atoms left after filtering, and :math:`k = 14.3996451` V·Å/e is the Coulomb constant. ``point_mag`` reports :math:`\lvert \mathbf{E} \rvert`. The sum has no cutoff or softening and is computed in single precision by the C extension.

We recommend filtering out the residue that contains ``center``, so that residue's own charges don't contribute to the field at its atoms. If ``center`` sits exactly on an atom, as with ``"method": "first"``, that atom's own charge would also turn the result into NaN. Example ``1A_point-field`` does this for residue CF1 378:

.. code-block:: json

    "filter_intersect": true,
    "filter": {
        "resid": ["CF1"],
        "resnum": [378]
    }

By default the field is given in the input file's frame. To give it in a frame aligned to the active site, add both ``x`` and ``y`` (see :doc:`Atom Specification <AtomSpec>`); giving ``x`` without ``y`` stops with an error. The magnitude doesn't depend on the frame.

Results go to a single file in ``outputpath``, with one line per structure:

- ``point_field.dat``: the x, y and z components, e.g. ``1_alcdehydro_run1:[-1.010131 0.5121924 0.28650615]``
- ``point_mag.dat``: the magnitude, e.g. ``1_alcdehydro_run1:1.1682429313659668``

Examples ``1A_point-field`` (input frame) and ``1B_point-field-oriented`` (oriented along a C=O bond) compute the same field both ways; the magnitudes agree.

Point calculations run in serial: structures are processed one at a time, and the sum over charges runs on a single thread. Unlike volume and topology runs, several copies of ``cpet.py`` can't share a folder, since each run rewrites the output file.

Volume E-field/ESP
---------------------------------------------------------
To be added.

Electric Field Topologies (Distribution of Streamlines)
--------------------------------------------------------
To be added. Advised to use parallel/GPU accelerated calculations.
