Parallel Execution
==================

Default Execution
-----------------

By default, PyCPET runs in serial mode. Executions over entire directories of input pdb/pqr are done by default, so, if care is taken to ensure file integrity,
users can instantiate cpet.py several times over different or the same directory to grossly parallelize calculations. This is generally not advised to speed up calculations,
and we encourage users to read below about specific parallelization options.

Accelerating Field Topology Calculations
----------------------------------------

Field topologies can get pretty expensive, particularly if many streamlines or very fine streamlines are chosen. There are both CPU (good for many streamlines) and GPU (good for many charges and/or fine streamlines)
options to speed up these calculations. Here is how to enable these options:

CPU Acceleration (through multiprocessing.Pool)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To enable CPU-accelerated field topology calculations, add the following to your options file:
{
    ...
    "concur_slip": N,
    ...
}

where N is the number of processes to use. This will parallelize the streamline calculations over N processes, which is generally good for many streamlines (e.g. >1000). Example benchmarking can be found in our 
paper [Ajmera2025]_. This is designed for HPC or more powerful workstations with many cores. The "topo" method by default uses parallelization, so you do not need to change that part of the options file.

GPU Acceleration (through pytorch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To enable GPU-accelerated field topology calculations, add the following to your options file:
{
    ...
    "method": topo_GPU,
    ...
}

This will **only use a single GPU** for acceleration. This is generally good for many charges (e.g. >100000) and/or fine streamlines (e.g. <0.01 Angstrom). Example benchmarking can be found in our paper [Ajmera2025]_. This is designed for HPC or workstations with a CUDA-capable GPU and the appropriate drivers installed.

Accelerating Electric Field Calculations
----------------------------------------

To be added.

