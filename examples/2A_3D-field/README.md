# Oriented point electric fields

This example determines the 3-D electric field electric field surrounding the midpoint of a C=O bond in a CXF residue, in a mutated alcohol dehydrogenase. Here, the field axes are oriented along the C=O bond as the x axis, with a nearby nitrogen as the y-axis. The options files has dimensions '0.5,0.5,0.5', indicating 0.5Å in the +-x, +-y, and +-z directions. Electric field output from two frames (as pdbs) is in the directory "outdir", each as its own file with transformation matrix and field information.

Run this example by simply "cpet.py >& cpet.out" or "cpet.py -o options/options.json >& cpet.out"

The output file has the 'density' which is the number of points along each dimension, and 'box volume', which are the half-lengths previously mentioned. Center and basis matrix information are stored here as well (for visualization purposes).