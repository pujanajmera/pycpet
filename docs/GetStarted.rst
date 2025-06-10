Getting Started
=================

Examples for pycpet are located in the `examples` directory of the repository. The examples are designed to demonstrate how to use the package for various applications.
Here, we will provide a fairly in-depth overview of *PyCPET*'s features, and how to use them.

PyCPET functionalities
----------------------
*PyCPET* is designed for the high throughput computation of electrostatic properties at enzyme active sites, including their dynamic behavior. This includes things like:
- Computing point electric fields/electrostatic potentials at arbitrary points, and over dynamic trajectories
- Computing fields/potentials in volumes, and their topological features defined by user input
- Clustering fields/potentials to get representative structures and understand the fluctuation of these quantities
- PCA on electric fields, useful for comparing fields across different enzyme classes
- Residue breakdown analysis of classical electric field topologies
- (Coming Soon) Electrostatic interaction energies between protein scaffold and active site charge densities
- And more!

We outline the tested functionalities below, primarily focusing on making your own "options" file for the cpet.py executable.

The Options File
-----------------
The options file is the cornerstone of *PyCPET*. It is a .json file that contains all relevant parameters you wish to specify for the calculation. Examples can be found in `examples`

General Options File Parameters
-------------------------------------------------
These are some general options that are required for most/all calculations.
- 'method': The type of calculation to run. The following methods are available:
    - 'point_field'
    - 'point_mag'
    - 