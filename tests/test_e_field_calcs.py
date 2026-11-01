import numpy as np
import pytest

from CPET.utils.calculator import (
    calculate_electric_field_base,
    calculate_electric_field_c_shared_full_alt,
    calculate_electric_field_gpu_for_test,
    calculate_field_at_point,
    compute_field_on_grid,
)

K = 14.3996451  # V * Angstrom / e


def reference_field(point, x, Q):
    """Coulomb field written out directly, in double precision."""
    R = point - x
    r = np.linalg.norm(R, axis=1, keepdims=True)
    return K * (Q * R / r ** 3).sum(axis=0)


@pytest.fixture(scope="module")
def charges():
    """40 random charges in a 12 A cube, and 5 points near its center."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-6.0, 6.0, (40, 3))
    Q = rng.uniform(-1.0, 1.0, (40, 1))
    points = rng.uniform(-1.0, 1.0, (5, 3))
    return x, Q, points


def test_single_charge():
    """One charge q at distance r gives k * q / r^2, pointing away from +q."""
    x = np.array([[0.0, 0.0, 0.0]])
    Q = np.array([[2.0]])

    E = calculate_field_at_point(x, Q, x_0=np.array([3.0, 0.0, 0.0]))

    np.testing.assert_allclose(E, [K * 2.0 / 9.0, 0.0, 0.0])


def test_field_between_opposite_charges_adds():
    """Midway between +q and -q, both charges push the field towards the negative one."""
    x = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    Q = np.array([[1.0], [-1.0]])

    E = calculate_field_at_point(x, Q, x_0=np.zeros(3))

    np.testing.assert_allclose(E, [2 * K, 0.0, 0.0])


def test_superposition(charges):
    """The field of all the charges is the sum of the fields of each one."""
    x, Q, points = charges
    one_by_one = sum(
        calculate_field_at_point(x[i : i + 1], Q[i : i + 1], x_0=points[0])
        for i in range(len(x))
    )

    np.testing.assert_allclose(
        calculate_field_at_point(x, Q, x_0=points[0]), one_by_one, rtol=1e-10
    )


def test_point_field_matches_reference(charges):
    x, Q, points = charges
    for p in points:
        np.testing.assert_allclose(
            calculate_field_at_point(x, Q, x_0=p), reference_field(p, x, Q), rtol=1e-10
        )


def test_c_field_matches_reference(charges):
    """The C extension works in single precision."""
    x, Q, points = charges
    for p in points:
        np.testing.assert_allclose(
            calculate_electric_field_c_shared_full_alt(p, x, Q),
            reference_field(p, x, Q),
            rtol=1e-4,
            atol=1e-5,
        )


def test_torch_field_matches_reference(charges):
    """The PyTorch implementation, on the CPU, agrees with the C extension."""
    x, Q, points = charges
    for p in points:
        np.testing.assert_allclose(
            calculate_electric_field_gpu_for_test(p, x, Q, device="cpu"),
            reference_field(p, x, Q),
            rtol=1e-4,
            atol=1e-5,
        )


def test_grid_field_matches_reference(charges):
    """compute_field_on_grid returns rows of x y z Ex Ey Ez."""
    x, Q, points = charges
    grid = points.reshape(5, 1, 1, 3)

    field = compute_field_on_grid(grid, x.copy(), Q.copy())

    assert field.shape == (5, 6)
    np.testing.assert_allclose(field[:, :3], points)
    expected = np.array([reference_field(p, x, Q) for p in points])
    np.testing.assert_allclose(field[:, 3:], expected, rtol=1e-4, atol=1e-5)


def test_field_magnitude_is_rotation_invariant(charges):
    """Rotating the charges and the point together doesn't change |E|."""
    x, Q, points = charges
    theta = 0.7
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    E = calculate_field_at_point(x, Q, x_0=points[0])
    E_rotated = calculate_field_at_point(x @ rotation.T, Q, x_0=rotation @ points[0])

    np.testing.assert_allclose(np.linalg.norm(E_rotated), np.linalg.norm(E))
    np.testing.assert_allclose(E_rotated, rotation @ E, atol=1e-10)


def test_point_on_a_charge():
    """A point exactly on a charge gives NaN; the grid calculation clamps r^2 instead."""
    x = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    Q = np.array([[1.0], [1.0]])
    p = np.zeros(3)

    assert np.isnan(calculate_electric_field_c_shared_full_alt(p, x, Q)).all()
    assert np.isfinite(compute_field_on_grid(p.reshape(1, 1, 1, 3), x, Q)).all()


@pytest.mark.xfail(
    strict=True,
    reason="calculate_electric_field_base divides by the norm of the whole R matrix, "
    "and returns one row per charge, so it is only right for a single charge",
)
def test_base_field_sums_to_reference(charges):
    x, Q, points = charges

    E = calculate_electric_field_base(points[0], x, Q)

    np.testing.assert_allclose(E.sum(axis=0), reference_field(points[0], x, Q), rtol=1e-6)
