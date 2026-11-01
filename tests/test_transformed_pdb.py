import numpy as np
import pytest

from CPET.source.calculator import calculator

WITH_AXES = {
    "CPET_method": "topo",
    "center": {"method": "first", "atoms": [{"CD": 2}]},
    "x": {"method": "mean", "atoms": [{"CG": 1}, {"CB": 1}]},
    "y": {"method": "mean", "atoms": [{"CA": 3}, {"CB": 3}]},
    "filter": {"resid": ["GLN"], "resnum": [2]},
    "filter_intersect": True,
    "filter_radius": 6.0,
    "n_samples": 8,
    "dimensions": [1.0, 1.0, 1.0],
    "step_size": 0.05,
    "concur_slip": 1,
    "initializer": "uniform",
    "max_streamline_init": "fixed_rand",
    "dtype": "float32",
}
NO_AXES = {
    "CPET_method": "point_field",
    "center": {"method": "first", "atoms": [{"CD": 2}]},
}
SHIFT = [0.5, 0.2, -0.3]


def read_pdb(path):
    """Coordinates of each atom, by atom number."""
    atoms = {}
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms[int(line[6:11])] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return atoms


@pytest.fixture
def transformed(tmp_path, test_files, monkeypatch):
    """Write transform_test_large.pdb for the given options and return what is needed to check it."""
    monkeypatch.chdir(tmp_path)
    pdb = test_files / "test_large.pdb"
    original = read_pdb(pdb)

    def run(options, strip_filter, box_shift=(0.0, 0.0, 0.0)):
        options = dict(
            options,
            write_transformed_pdb=True,
            strip_filter=strip_filter,
            box_shift=list(box_shift),
        )
        calc = calculator(options, path_to_pdb=str(pdb))
        written = read_pdb(tmp_path / "transform_test_large.pdb")
        return calc, written, original

    return run


def expected(calc, original, serial, rotate, box_shift):
    """Where an atom should end up: centered, rotated into the box frame, protein shifted."""
    x = original[serial] - calc.center
    if rotate:
        x = x @ np.linalg.inv(calc.transformation_matrix)
    return x - np.array(box_shift)


@pytest.mark.parametrize("strip_filter", [False, True])
@pytest.mark.parametrize("box_shift", [(0.0, 0.0, 0.0), tuple(SHIFT)])
def test_atoms_are_written_in_the_box_frame(transformed, strip_filter, box_shift):
    """Each atom keeps its own number and gets its own transformed coordinates.

    With strip_filter, the atoms used to get the numbers of other atoms, and without it
    the protein was shifted the wrong way.
    """
    calc, written, original = transformed(WITH_AXES, strip_filter, box_shift)

    for serial, coords in written.items():
        np.testing.assert_allclose(
            coords, expected(calc, original, serial, True, box_shift), atol=2e-3
        )


def test_strip_filter_keeps_only_the_charges_used(transformed):
    _, all_atoms, original = transformed(WITH_AXES, strip_filter=False)
    calc, kept, _ = transformed(WITH_AXES, strip_filter=True)

    assert len(all_atoms) == len(original)
    assert len(kept) == len(calc.x) < len(all_atoms)


def test_both_modes_agree_on_the_atoms_they_share(transformed):
    _, all_atoms, _ = transformed(WITH_AXES, strip_filter=False, box_shift=SHIFT)
    _, kept, _ = transformed(WITH_AXES, strip_filter=True, box_shift=SHIFT)

    assert set(kept) <= set(all_atoms)
    for serial, coords in kept.items():
        np.testing.assert_allclose(coords, all_atoms[serial], atol=2e-3)


@pytest.mark.parametrize("strip_filter", [False, True])
def test_without_axes_the_structure_is_only_centered(transformed, strip_filter):
    """With no x and y there is no rotation. This used to raise an AttributeError."""
    calc, written, original = transformed(NO_AXES, strip_filter)

    assert len(written) > 0
    for serial, coords in written.items():
        np.testing.assert_allclose(
            coords, expected(calc, original, serial, False, (0, 0, 0)), atol=2e-3
        )
