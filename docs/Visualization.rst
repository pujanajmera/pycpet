Visualization
=================

Electric Fields
---------------

``visualize_field`` turns each field file (``*_efield.dat``) in ``inputpath`` into a ChimeraX ``.bild`` file of arrows in ``outputpath``, with the same name. The arrows are placed back in the input file's frame, so they line up with the original structure: open both in ChimeraX (``open protein.pdb``, then ``open protein_efield.bild``).

- Each arrow sits on a grid point and points along the field. Longer, bluer arrows are stronger; short, white ones are weaker.
- Magnitudes are rescaled from 0 to 1 within each file, so colors and lengths can't be compared between files. The log prints each file's maximum and minimum field magnitude (in V/Å), which set that scale.

Two options, under ``visualization``, control what's drawn:

- ``cutoff``: a percentile (0–100) of field magnitude; weaker arrows aren't drawn. Raise it to show only the strongest parts of a busy field.
- ``sparsify_factor``: draws every Nth grid point along each axis, for fewer arrows. It's rounded to a factor shared by the number of points along x, y and z, so if any axis has a prime number of points (such as 11) it has no effect.

Example ``2B_3D-field-visualization`` draws the fields from ``2A_3D-field``. Visualization runs in Python.

Electrostatic Potentials
------------------------

``visualize_esp`` turns each potential file (``*_esp.dat``) in ``inputpath`` into a ``.bild`` file of spheres in ``outputpath``, with the same name and in the input file's frame, as for fields.

- Each sphere sits on a grid point. Positive potentials go from white to red and negative ones from white to blue, so the strongest are the most saturated.
- Sphere size grows with the size of the potential, up to about half the grid spacing.
- ``cutoff`` works as for fields, on the absolute potential. ``sparsify_factor`` has no effect.

By default, each sign gets its own color scale within each file, and the log prints its range (in V):

.. code-block:: text

    Positive ESP max: 1.536, min: 0.001
    Negative ESP max: -0.001, min: -0.301

To compare files on a common scale, set the limits yourself with ``max_p_esp`` and ``min_p_esp`` (most and least positive), and ``max_n_esp`` and ``min_n_esp`` (least and most negative, so ``max_n_esp`` is the one nearest zero). Choose limits that cover every file: potentials outside them aren't clipped, and give invalid colors.

Drawing the Box in ChimeraX
---------------------------

ChimeraX draws boxes only along its own axes, so a box aligned to ``x`` and ``y`` can't be drawn over the original structure. Instead, *PyCPET* can write the structure in the box's frame, where the box is centered on the origin and aligned with the axes.

1. Add ``"write_transformed_pdb": true`` to the options file of any point, volume or topology run that sets ``x`` and ``y``. Each structure is then also written to ``transform_<name>.pdb``, in the folder you run ``cpet.py`` from rather than ``outputpath``.

   In this file, ``center`` is at the origin, the ``x`` point lies on the +x axis and the ``y`` point in the xy-plane, on the +y side. Atoms keep their names and residue numbers, but every atom gets chain A.

2. Write a ``box.bild`` file (see the `BILD format <https://www.cgl.ucsf.edu/chimerax/docs/user/formats/bild.html>`_) whose corners are minus and plus ``dimensions``. For ``"dimensions": [0.5, 0.5, 0.5]``:

   .. code-block:: text

       .transparency 0.5
       .color blue
       .box -0.5 -0.5 -0.5 0.5 0.5 0.5

3. Open both in ChimeraX: ``open transform_protein.pdb``, then ``open box.bild``.

By default the file keeps every atom, including the filtered ones, so the residue at ``center`` still shows. ``"strip_filter": true`` leaves out every atom removed by filtering, including those outside ``filter_radius``.

.. note::

   Field and potential ``.bild`` files from ``visualize_field`` and ``visualize_esp`` are in the original frame, so they line up with the original structure, not the transformed one. With a nonzero ``box_shift``, the structure in the transformed file is shifted the opposite way, so the box drawn around the origin stays in the right place.

*Figure to be added: a transformed structure with its box, drawn in ChimeraX.*
