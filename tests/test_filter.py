import numpy as np
import pytest

from CPET.utils.io import (
    filter_atomspec,
    filter_in_box,
    filter_radius,
    filter_radius_whole_residue,
)

CENTER = np.zeros(3)


def test_filter_radius(synthetic_data):
    """Only charges closer than the radius to the center are kept."""
    x, Q = synthetic_data["x"], synthetic_data["Q"]

    x_filt, Q_filt = filter_radius(x, Q, CENTER, radius=2.0)

    # distances: 0, 1.5, 2.33, 8.66
    np.testing.assert_allclose(x_filt, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    np.testing.assert_allclose(Q_filt, [[1.0], [-0.5]])


def test_filter_radius_whole_residue(synthetic_data):
    """Residues with no atom inside the radius are removed entirely."""
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]

    x_filt, Q_filt, ID_filt = filter_radius_whole_residue(
        x, Q, ID, CENTER, radius=2.0
    )

    # WAT 10 is inside; FE 651 (2.33 A) and NI 652 (8.66 A) are not
    np.testing.assert_allclose(x_filt, x[:2])
    np.testing.assert_allclose(Q_filt, Q[:2])
    assert [row[3] for row in ID_filt] == [10, 10]


def test_filter_radius_whole_residue_keeps_outside_atoms(synthetic_data):
    """One atom inside the radius keeps the whole residue, even atoms outside."""
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]

    # Only the first atom (0 A) is within 1 A; the second WAT 10 atom is at 1.5 A
    x_filt, _, ID_filt = filter_radius_whole_residue(x, Q, ID, CENTER, radius=1.0)

    np.testing.assert_allclose(x_filt, x[:2])
    assert [row[1] for row in ID_filt] == ["O", "H"]


def test_filter_atomspec_union(synthetic_data):
    """Without intersect, an atom matching any listed value is removed."""
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]

    x_filt, Q_filt, ID_filt = filter_atomspec(
        x, Q, ID, {"atom_type": ["FE"], "resid": ["NI"]}, intersect=False
    )

    assert [row[1] for row in ID_filt] == ["O", "H"]
    np.testing.assert_allclose(x_filt, x[:2])
    assert len(Q_filt) == 2


def test_filter_atomspec_intersect(synthetic_data):
    """With intersect, an atom must match every key of a filter entry."""
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]

    # Residue WAT 10 only: matches both of its atoms, nothing else
    x_filt, _, ID_filt = filter_atomspec(
        x, Q, ID, {"resid": ["WAT"], "resnum": [10]}, intersect=True
    )
    assert [row[0] for row in ID_filt] == [2001, 3001]

    # Atom H in WAT 10 only
    _, _, ID_filt = filter_atomspec(
        x, Q, ID, {"atom_type": ["H"], "resid": ["WAT"]}, intersect=True
    )
    assert [row[0] for row in ID_filt] == [1001, 2001, 3001]


def test_filter_atomspec_unknown_key(synthetic_data):
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]
    with pytest.raises(ValueError, match="not recognized"):
        filter_atomspec(x, Q, ID, {"residue": ["WAT"]}, intersect=False)


def test_filter_atomspec_intersect_needs_equal_lengths(synthetic_data):
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]
    with pytest.raises(ValueError, match="same length"):
        filter_atomspec(
            x, Q, ID, {"resid": ["WAT", "FE"], "resnum": [10]}, intersect=True
        )


def test_filter_in_box(synthetic_data):
    """Charges inside the box are removed; those outside are kept."""
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]
    center = np.array([1.0, 0.0, 0.0])

    x_filt, Q_filt, ID_filt = filter_in_box(x, Q, ID, center, [1.5, 1.5, 1.5])

    # Relative to the center, the first three atoms lie within +-1.5 A on every axis
    np.testing.assert_allclose(x_filt, [[5.0, 5.0, 5.0]])
    np.testing.assert_allclose(Q_filt, [[1.2]])
    assert ID_filt[0][1] == "NI"


def test_filter_in_box_is_strict(synthetic_data):
    """A charge exactly on the box boundary is not inside it."""
    x, Q, ID = synthetic_data["x"], synthetic_data["Q"], synthetic_data["ID"]

    # Box edge at x = 1.5, exactly where the second atom sits
    x_filt, _, _ = filter_in_box(x, Q, ID, CENTER, [1.5, 1.5, 1.5])

    assert [1.5, 0.0, 0.0] in x_filt.tolist()
    assert [0.0, 0.0, 0.0] not in x_filt.tolist()
