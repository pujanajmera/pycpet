import numpy as np
import pytest

from CPET.utils.io import parse_pqr

# Atoms 10, 16492 and -10 of test_large.pqr
INDICES = [10, 16492, -10]
COORDS = np.array(
    [
        [76.062, 94.681, 129.367],
        [114.752, 113.583, 105.331],
        [110.242, 117.554, 92.352],
    ]
)
CHARGES = [-0.243, -0.101, 0.068]
RESIDS = ["ARG", "THR", "HEM"]
ATOM_TYPES = ["NH2", "CA", "H165"]
ATOM_NUMBERS = [11, 16493, 31843]


@pytest.fixture(scope="module")
def large_pqr(test_files):
    return parse_pqr(str(test_files / "test_large.pqr"))


def test_pqr_parser_shapes(large_pqr):
    x, Q, atom_number, resid, resnum, atom_type = large_pqr

    n = 31852
    assert x.shape == (n, 3)
    assert Q.shape == (n, 1)
    for column in (atom_number, resid, resnum, atom_type):
        assert column.shape == (n,)


def test_pqr_parser_coordinates_and_charges(large_pqr):
    x, Q, *_ = large_pqr

    np.testing.assert_allclose(x[INDICES], COORDS)
    np.testing.assert_allclose(Q[INDICES].ravel(), CHARGES)
    assert Q.max() < 1.0
    assert Q.min() > -1.0


def test_pqr_parser_identities(large_pqr):
    """Large atom and residue numbers, where columns run together, still parse."""
    _, _, atom_number, resid, _, atom_type = large_pqr

    assert list(resid[INDICES]) == RESIDS
    assert list(atom_type[INDICES]) == ATOM_TYPES
    assert list(atom_number[INDICES]) == ATOM_NUMBERS


def test_pqr_parser_small_file(tmp_path):
    pqr = tmp_path / "two_atoms.pqr"
    pqr.write_text(
        "ATOM      1  N   ARG A   1      78.012  97.787 135.684 -0.239  1.550\n"
        "HETATM    2  O   WAT A   2      10.000  20.500 -30.250  0.410  1.520\n"
        "TER\n"
    )

    x, Q, atom_number, resid, resnum, atom_type = parse_pqr(str(pqr))

    np.testing.assert_allclose(x, [[78.012, 97.787, 135.684], [10.0, 20.5, -30.25]])
    np.testing.assert_allclose(Q.ravel(), [-0.239, 0.410])
    assert list(atom_number) == [1, 2]
    assert list(resid) == ["ARG", "WAT"]
    assert list(resnum) == [1, 2]
    assert list(atom_type) == ["N", "O"]
