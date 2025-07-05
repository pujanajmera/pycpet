Special Scripts
=================

The following are tutorials for special scripts that aren't built into the cpet.py executable, but use the pycpet libraries to do some neat analyses.

Interaction Energy Analysis
-------------------------------------------------
Authors: Anubhav Goswami, Pujan Ajmera. Credit to the Binju Wang group for the original method (w/o nuclear correction): https://doi.org/10.1021/acs.jpcb.3c01054

The script is electrostatic_interaction_QM.py, and it computes interaction energies (for QM/MM calculations/QM calculations with point charge embedding) using these equation:

.. math::
    E_{int_elec} = \sum_{i=1}^{N} \int_{V} \frac{q_i \rho(\mathbf{r})}{|\mathbf{r} - \mathbf{r}_i|} d\mathbf{r}
    E_{int_nuc} = \sum_{i=1}^{N} \sum_{j=1}^{M} \frac{q_i Q_j}{r_{ij}}
    E_int = E_{int_elec} + E_{int_nuc}

where :math:`q_i` and :math:`Q_j` are the charges of MM region of interest and the nuclear charges of the QM density, respectively. These are the following options that can be passed to the script:

- `-m/--molden`: The molden input file, which contains the QM density in a molden format.
- `-p/--pdb`: The path to the a PDB file that contains charges of all MM atoms.
- `-r/--res`: Flag for residue breakdown. If provided, analysis will be done by residue as well.
- `-o/--options`: PyCPET options file. Please see the :doc:`Get Started <../GetStarted>` page for more information on how to write an options file. This is solely used in this script to filter out the QM region atoms, otherwise the interaction energy will be unrealistic.
- `-v/--verbose`: Flag for verbose output. If provided, the script will tell you what interaction it is computing.

These interaction energies are output to the command line, so the following usage is recommended (The script should be in your path if you have correctly installed PyCPET):

```bash
electrostatic_interaction_QM.py -m name_of_molden_file -p protein.pdb -o options.json -r
``` 

To see an example usage of this, check out this work that applies it to chorismate mutases: (CITE)


