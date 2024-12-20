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

## Examples

Several examples are in the ```examples``` directory. Most of these are designed for running high-throughput, but can readily be done for single calculations.
