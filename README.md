# *PyCPET*

Python-based Computation of Protein Electric Field Topology, built for high-throughput accelerated calculations of several electrostatic properties of enzyme active sites from simulations. This program is incredibly flexible and scriptable for virtually any analysis of classical electric fields and electrostatic potentials.

## Cite
To cite your use of *PyCPET*, please use the following:
Ajmera, P., Vargas, S., Chaturvedi, S., Hennefarth, M. & Alexandrova, A. *PyCPET* - Computing Heterogeneous 3D Protein Electric Fields and Their Dynamics. Journal of Chemical Theory and Computation (2025). doi:10.1021/acs.jctc.5c00138

## Requirements and Installation
System requirements:
- gcc (to compile C-shared libraries)
- anaconda (preferred, not required)

Follow these recommended steps to install PyCPET

1. Make a clean conda environment (recommended, not required)

conda create -n pycpet-test python=3.11 -y

2. Run pip install in the conda environment

pip install pycpet

3. When running, we advise you set the following:

CPU multithreading (available for topologies): 
    export OMP_NUM_THREADS=1
GPU-accelerated code (available for topologies, fields in dev):
    export CUDA_VISIBILE_DEVICES=N (where N is the GPU number, only needed for non-HPC setups)

## Documentation
Almost all use of *PyCPET* is either scripting with the objects provided (requires in-depth knowledge of the code) or using the cpet.py script with an options file. We are developing in-depth documentation of how to format and use the options file, examples, and cookbooks at (), but key features and explanation of example files are noted below.

## Features

These are the following available features, and their corresponding options file 'method' keywords:

- Computing point electric fields: 'point_field'
- Computing point electric field magnitudes: 'point_mag'
- Computing 3-D electric fields: 'volume'
- Computing 3-D electrostatic potentials: 'volume_ESP'
- Computing 3-D distribution of streamlines: 'topo' (CPU, default) and 'topo_GPU' (GPU)
- Clustering by distribution of streamlines: 'cluster'
- Clustering by 3-D electric field (tensor decomp): 'cluster_volume_tensor' **IN BETA**
- Clustering by 3-D electrostatic potential: 'cluster_volume_ESP_tensor' **IN BETA**
- Visualizing 3-D fields: 'visualize_fields'
- PCA on 3-D fields, for a single set of data: 'pca' **IN BETA**
- PCA on 3-D fields, for a full comparative analysis between multiple sets of field data (e.g. variants): 'pca_compare' **IN BETA**
- Finding if any atoms are intruding the field volume: 'box_check'

## Examples

Several examples are in the ```examples``` directory. Most of these are designed for single calculations, but can be extended to high-throughput with almost no changes.

## Specialized Scripts

For features unavailable from the cpet.py script mentioned above, we offer scripts in ```source/scripts``` for the following. Please note that these are not rigorously tested for all cases, but showcase the scripting ability of the pycpet library:

- residue_breakdown_analysis.py: Residue contribution to topology over dynamics, ranked. Requires completed topology calculations for an MD **IN BETA**
- tensor_based_cluster_double.py: Electrostatic potential/electric field-based clustering for two sets of electric fields/electrostatic potentials. Assumes that all field calculations have been completed for both sets of directories. **IN BETA**

## Note: Formatting of PDB/PQR files

Most of the code here requires well-formated PDB or PQR files. The formatting is as follows (see io.py for more details):

PDB:
- Up to and including charge information, follow standard formatting: https://www.cgl.ucsf.edu/chimerax/docs/user/formats/pdbintro.html
- For charge information, charges must be in columns 55-64 of each line, inclusive

PQR:
- 
