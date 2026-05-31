Visualization
=================

Electric Fields
-----------------
To be added.

Electrostatic Potentials
---------------------------------------------------------
To be added.

Using `write_transformed_pdb` for Chimera/ChimeraX Electric Field Schemes
------------------------------------------------------------------------------------
Chimera/ChimeraX unfortunately can only make boxes that are oriented in the softwares global frame, which is where `write_transformed_pdb` comes in handy.
By using `write_transformed_pdb: True` and `strip_filter: False` in the options file along with writing a `.bild <https://www.cgl.ucsf.edu/chimerax/docs/user/formats/bild.html>`_ file as follows:

.. code-block:: console

    .transparency 0.5
    .color blue
    .box x1 y1 z1 x2 y2 z2

, replacing x1 y1 z1 and x2 y2 z2 with the negative of the `dimensions` option and the `dimensions` option itself, you can get great schemes like this to show your volume calculation:

To be added.

API Examples for other Data Visualizations
--------------------------------------------

It may be of interest, for example, to visualize field topologies, or observe the distribution of point fields across a trajectory.
The API provided through pycpet makes this pretty straightforward.

Examples here to be added.
