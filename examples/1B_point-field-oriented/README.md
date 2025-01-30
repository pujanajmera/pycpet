# Oriented point electric fields

This example determines the point electric field at the midpoint of a C=O bond in a CXF residue, in a mutated alcohol dehydrogenase. Here, the field axes are oriented along the C=O bond as the x axis, with a nearby nitrogen as the y-axis. These two axes form a transformation matrix. Electric field output from two frames (as pdbs) is in the directory "outdir".

Run this example by simply "cpet.py >& cpet.out" or "cpet.py -o options/options.json >& cpet.out"

Can compare with 'point-field' example to see how the field magnitude does not change under a transformation.