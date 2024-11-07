# pycpet

Python-based Classical Protein Electric Field Topology, built for high-throughput accelerated calculations of several electrostatic properties of enzyme active sites from simulations.

System requirements:
- gcc (if you want to optimize speed)
- OTHER STUFF?

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
    export OMP_NUM_THREADS=0
GPU-accelerated code:
    




Installing:

pip install -e .

Upon installing, you will see the executable cpet.py available. This is the operating script for all calculations. An example options file is provided under source/options, and more details about the available options are below

For pycuda you might need:

export CPATH=$CPATH:/usr/local/cuda/include

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

For c-shared libraries manually setting the number of OMP threads is needed: 

export OMP_NUM_THREADS=0
