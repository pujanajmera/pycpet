# Oriented point electric fields

This example uses the 3-D electric field electric field determined in the 3D-field-visualization example, surrounding the midpoint of a C=O bond in a CXF residue, in a mutated alcohol dehydrogenase. Here, the field axes are oriented along the C=O bond as the x axis, with a nearby nitrogen as the y-axis. **Only the field files (*_efield.dat, the default output of volume field calculations) are needed in the input directory.**

Some of the options here include 'sparsification' and 'cutoff'. The former, which just sparisifies the grid output compared to the input field, is more useful, with a higher number indicating a more sparse grid. The latter, is a percentile cutoff - with 0 indicating the full field. This is useful for complex fields with multiple local maxima upon initial visualization.

Run this example by simply "cpet.py >& cpet.out" or "cpet.py -o options/options.json >& cpet.out"

The output file is a '.bild.' format, which is compatible with Chimera/ChimeraX, and intended to overlay on the original pdb file.