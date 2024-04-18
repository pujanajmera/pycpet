# CPET-python

Classical Protein Electric Field Topology (https://github.com/matthew-hennefarth/CPET) but in python, and built for high-throughput accelerated calculations

Original code in C++ is by Matthew Hennerfarth

Installing:

pip install -e .

Upon installing, you will see the executable cpet.py available. This is the operating script for all calculations. An example options file is provided under source/options, and more details about the available options are below

For pycuda you might need:

export CPATH=$CPATH:/usr/local/cuda/include

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

For c-shared libraries manually setting the number of OMP threads is needed: 

export OMP_NUM_THREADS=0
