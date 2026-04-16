.. pycpet documentation master file, created by
   sphinx-quickstart on Thu Feb 13 20:28:15 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycpet's documentation!
==================================
PyCPET, or Python-based Computation of Protein Electric Field Topology, is built for high-throughput accelerated calculations of several electrostatic properties of enzyme active sites from simulations. This program is flexible and scriptable for analysis of classical, quantum mechanical, or QM/MM electric fields and electrostatic potentials.

Key features:

- Computing point and 3D grid-based electric fields and electrostatic potentials
- Active site volume definitions and orientation by flexible selection of atoms/coordinates
- High-throughput distribution of streamlines for electric field topologies, with established benchmarking protocols
- High-throughput clustering molecular dynamics trajectories by either histograms of field streamlines or tensor decomposition of fields/electrostatic potentials. GPU-acceleration available.
- QM/MM electrostatic breakdown analysis, compatible with ORCA, Turbomole, and PySCF

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   GetStarted
   AtomSpec
   FieldTopESP
   Visualization
   Dynamics
   SpecialScripts
   Parallel
   Example Suite
   citation
   apiref



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
