Input Files
===========

*PyCPET* reads structures as PDB or PQR files that carry partial charges. Set ``inputfiletype`` to ``"pdb"`` (the default) or ``"pqr"``; every file of that type in ``inputpath`` is used. Only ``ATOM`` and ``HETATM`` lines are read.

Output files are named after the input file, up to its first dot: ``frame1.pdb`` gives ``frame1.top``, ``frame1_efield.dat`` and so on.

PDB
---

- Follow the `standard PDB format <https://www.cgl.ucsf.edu/chimerax/docs/user/formats/pdbintro.html>`_ up to the coordinates.
- Charges go in columns 56–64 of each line.
- Chain IDs are read from column 22, so ``chain`` filters and ``chains`` selections work (see :doc:`Atom Specification <AtomSpec>`).

PQR
---

- Lines must include a chain ID column; lines without one fail to parse. Chain IDs are not kept, so ``chain`` filters and ``chains`` selections don't apply to PQR input.
- Coordinates must be in columns 31–54 and charges in columns 55–61. For residue numbers above 999, these columns shift right by one for each extra digit.
- Radii are not read.
