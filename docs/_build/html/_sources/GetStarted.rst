Getting Started - Input Options File
=========================================

Examples for pycpet are located in the `examples` directory of the repository. The examples are designed to demonstrate how to use the package for various applications.
Here, we will provide a fairly in-depth overview of *PyCPET*'s features, and how to use them.

PyCPET features
----------------------
*PyCPET* is designed for the high throughput computation of electrostatic properties at enzyme active sites, including their dynamic behavior. This includes:

**Point Electrostatic Analysis + Dynamics**

- Computing point electric fields/electrostatic potentials at user-specified points, over dynamic trajectories

**Volume Electrostatic Analysis + Dynamics**

- Computing fields/potentials in volumes, and their topological features defined by user input (parallelized!)
- Clustering fields/potentials to get representative structures and dynamic fluctuations (parallelized!)
- Principal Component Analysis (PCA) on 3D electric fields, useful for comparing fields across enzymes
- Residue breakdown analysis of classical electric field topologies over dynamics

**Quantum Mechanical Electrostatic Analysis**

- Electrostatic interaction energies between protein scaffold and active site charge densities

We outline the tested functionalities below, primarily focusing on making your own "options" file for the cpet.py executable.

The Options File
-----------------
The options file is the cornerstone of *PyCPET*. It is a .json file that contains all relevant parameters you wish to specify for the calculation. Examples can be found in `examples`

General Options File Parameters
-------------------------------------------------
These are some general options that are required for most/all calculations.

1. `CPET_method`: The method to use for the calculation. These are the tested available methods:
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

2. `input_path`: The path to input structures, typically a folder of pdb/pqr files or a folder of 3D electric field/topology files

3. `output_path` (optional): The path to output the results. If not specified, results will be output to a default directory typically `./cpet/`

4. `write_transformed_pdb` (optional): If True, write out the transformed pdb files after orienting to the active site frame. Default is False. This function is useful for making schemes in ChimeraX, see :doc:`Visualization <../Visualization>`.

5. `strip_filter` (optional): If True while `write_transformed_pdb` is True, the transformed pdb files will be stripped of filtered residue.

Field Options File Parameters
-------------------------------------------------
These are some general options that are required for electric field, or electrostatic potential calculations.

1. `center`: The center of the calculation. For point fields/electrostatic potential, this is where the quantity is computed. See :doc:`AtomSpec <../AtomSpec>` for more information.

2. `x`: The point used for defining the x-axis of the volume. See :doc:`AtomSpec <../AtomSpec>` for more information.

3. `y` (optional): The point used for defining the y-axis of the volume. See :doc:`AtomSpec <../AtomSpec>` for more information. If this isn't provided, there will be no rotation into the active site frame.

4. `dimensions` (optional): The **half width** dimensions of the volumes to compute 3D fields/topology/ESP, provided as a list of 3 numbers, e.g. `[0.5,0.5,0.75]` is a 1x1x1.5 volume. Units are in Angstroms.

5. `step_size` (optional): The grid size used for computing 3D fields/ESP, provided as a single number. For topologies, this number dictates how the streamlines are seeded and the steps size of the streamline itself and is a convergence parameter. Units are in Angstroms.

6. `n_samples` (optional): Only required for 3D field topologies, this is the integer number of streamlines to compute, and acts as a convergence parameter. See :doc:`Parallel <../Parallel>` for more information on how to parallelize over streamlines.

Filtering parameters:

7. `filter_intersect` (optional): If True, find the intersection of the provided filters. If False, find the union of the provided filters. Default is False.

8. `filter` (optional): A list of residue/atom/etc filters to apply to the field/topology/ESP calculations. See :doc:`AtomSpec <../AtomSpec>` for more information.

9. `filter_radius` (optional): If specified, the radius from the `center` selection to exclude atoms/residues from the field/topology/ESP calculations. Units are in Angstroms.

Clustering Options File Parameters
-------------------------------------------------
TBD

Visualization Options File Parameters
-------------------------------------------------
TBD




