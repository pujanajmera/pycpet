# CPET-python

Classical Protein Electric Field Topology (https://github.com/matthew-hennefarth/CPET) but in python, and built for high-throughput accelerated calculations

Original code in C++ is by Matthew Hennerfarth

To do items:

- Finish field calculation
- Improve options parsing
- Accelerate topology calcs
- Try gridding topology calcs
- Add cylinder and sphere region options
- Test invariances

For pycuda you might need:

export CPATH=$CPATH:/usr/local/cuda/include

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
