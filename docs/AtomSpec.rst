Atom Specification
=====================

Atom Specification for Volume Orientation
---------------------------------------------

``center``, ``x`` and ``y`` each take either fixed coordinates or a selection of atoms.

**Fixed coordinates:** a list of three numbers, in the input file's frame:

.. code-block:: json

    "center": [10.2, -4.5, 33.0]

**Atom selection:** a ``method`` and a list of ``atoms``. Each entry in ``atoms`` maps an atom name to a residue number:

.. code-block:: json

    "center": {
        "method": "mean",
        "atoms": [
            {"C7": 378},
            {"O9": 378}
        ]
    }

``method`` turns the selected atoms into one point:

- ``"mean"``: the average position of the selected atoms.
- ``"first"``: the position of the first selected atom.
- ``"inverse"``: the first selected atom reflected through the average of the others (2 x average of the others - first).

Add ``chains``, a list the same length as ``atoms``, to also match on chain ID:

.. code-block:: json

    "x": {
        "method": "first",
        "atoms": [{"O9": 378}],
        "chains": ["A"]
    }

Without ``chains``, "first" means the first selected atom in the order of the input file. With ``chains``, it means the first entry in ``atoms``.

The axes are built from these points: ``x`` sets the direction of the x-axis from ``center``, and ``y`` fixes the y-axis. The two points must differ. Without ``x``, the input file's frame is used and ``y`` is ignored. ``volume``, ``volume_ESP``, ``topo`` and ``topo_GPU`` need both ``x`` and ``y``.

Atom Specification for Filtering in Field/ESP Calculations
-------------------------------------------------------------

``filter`` removes atoms from the charges used in the calculation. Its keys are ``atom_number``, ``atom_type`` (atom name), ``resid`` (residue name), ``resnum`` and ``chain``, and each takes a list of values.

By default (``"filter_intersect": false``), an atom is removed if it matches **any** value of **any** key. This removes every atom of residue 378 and every ``WAT`` residue:

.. code-block:: json

    "filter": {
        "resnum": [378],
        "resid": ["WAT"]
    }

With ``"filter_intersect": true``, the lists are read position by position, and an atom is removed only if it matches every key at that position. All lists must be the same length, and ``"*"`` or ``""`` matches anything. This removes residue ``CF1`` 378, and atom ``FE`` of residue 401 in any residue name:

.. code-block:: json

    "filter_intersect": true,
    "filter": {
        "resid": ["CF1", "*"],
        "resnum": [378, 401],
        "atom_type": ["*", "FE"]
    }

Two more filters work by position rather than identity:

- ``filter_radius``: keeps only residues with at least one atom within this distance (in Angstroms) of ``center``, and removes whole residues outside it.
- ``filter_in_box``: removes charges inside the box defined by ``dimensions``.

See the :doc:`Options Reference <Options>` for defaults.
