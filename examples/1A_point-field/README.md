# Oriented point electric fields

This example determines the point electric field at the midpoint of a C=O bond in a CXF residue, in a mutated alcohol dehydrogenase. Here, the field axes ***are not*** oriented and is just in the global frame of reference. This is enabled by removing the y-point from the options.json file. Electric field output from two frames (as pdbs) is in the directory "outdir".

Run this example by simply "cpet.py >& cpet.out" or "cpet.py -o options/options.json >& cpet.out"
