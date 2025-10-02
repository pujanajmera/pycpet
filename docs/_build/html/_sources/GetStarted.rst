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
- Electrostatic interaction energies between protein scaffold and active site charge densities
- And more!

We outline the tested functionalities below, primarily focusing on making your own "options" file for the cpet.py executable.

The Options File
-----------------
The options file is the cornerstone of *PyCPET*. It is a .json file that contains all relevant parameters you wish to specify for the calculation. Examples can be found in `examples`

General Options File Parameters
-------------------------------------------------
These are some general options that are required for most/all calculations.

1. `method`: The method to use for the calculation. These are the tested available methods:
    - Standalone methods that can be run for single pdbs, or over any set of pdbs in a directory as long as they have a common orientable atom set:
    - `point_field`: Computing the electric field at a point, either arbitrary or defined by some atom selection.
    - `point_mag`: Computing the electric field magnitude at a point.
    - `volume`: Computing the electric field in a volume, defined by an oriented grid.
    - `volume_ESP`: Computing the electrostatic potential in a volume, defined by an oriented grid.
    - `topo`: Computing the electric field topology in a volume by distribution of streamlines (multithreaded). See https://doi.org/10.1021/acscatal.0c02795 and https://doi.org/10.1007/978-3-540-89689-0_23
    - `topo_GPU`: Same as above, but GPU-accelerated. See https://doi.org/10.1021/acs.jctc.5c00138
    - `box_check`: Prints out any residues that are intruding in the box volume defined by the user

    - The below are methods that depend on one or more previous calculations run above. These provide immense value for dynamics analysis.
    - `cluster`: Clustering a set of electric field topologies to get representative fields
    - `cluster_volume_tensor`: Clustering a set of electric field volumes to get representative fields, using a tensor-decomposition. See (CM Paper CITE)
    - `cluster_volume_esp_tensor`: Clustering a set of electrostatic potential volumes to get representative electrostatic potentials, using a tensor-decomposition. This is the only method to use for electrostatic potential clustering.
    - `pca`: Performing PCA on a set of electric field volumes, especially useful for comparing field dynamics between mutants. See https://doi.org/10.1021/jacs.4c03914 and https://doi.org/10.1021/acs.jctc.5c00138
    - `visualize_field` or `visualize_esp`: Make a .bild file for ChimeraX to visualize the electric field or electrostatic potential, respectively.
