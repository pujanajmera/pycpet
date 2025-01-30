# pycpet

Python-based Computation of Protein Electric Field Topology, built for high-throughput accelerated calculations of several electrostatic properties of enzyme active sites from simulations.

System requirements:
- gcc (if you want to optimize speed)

Follow these steps to install PyCPET

1. Clone the repo:

git clone https://github.com/pujanajmera/CPET-python.git

2. (Highly recommended but not required) Install the pre-made conda environment CPET_ENV in a path directory:

conda env create -f CPET_ENV.yml -p PATH_TO_ENVIRONMENT_LOCATION

3. In the current directory, do the following (if you are using CPET_ENV, make sure to be in 
the environment when running this step):

pip install .

(Developer version) pip install -e .

From here, the executable cpet.py will be available in your path (or in your environment if using conda)

4. To compile the C math modules, head over to CPET/utils and run:

gcc -fopenmp math_module.c -o math_module.so -shared -fPIC -O3 -march=native -funroll-loops -ffast-math

5. When running, we advise you set the following:

CPU multithreading: 
    export OMP_NUM_THREADS=1
GPU-accelerated code:
    export CUDA_VISIBILE_DEVICES=N (where N is the GPU number, only needed for non-HPC setups)

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


## Formatting

Most of the code here requires well-formated PDB or PQR files. The formatting is as follows (see io.py for more details):

PDB:
- Up to and including charge information, follow standard formatting: https://www.cgl.ucsf.edu/chimerax/docs/user/formats/pdbintro.html
- For charge information, charges must be in columns 55-64 of each line, inclusive

PQR:
- 