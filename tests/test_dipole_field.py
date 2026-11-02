import ctypes

import numpy as np
import pytest
from numpy.ctypeslib import ndpointer

from CPET.utils.calculator import Math

K = 14.3996451  # V * Angstrom / e


def kernel_field(point, x, mu):
    """Field at a point from the C kernel for point dipoles (calc_field_base_dipole)."""
    kernel = Math.math.calc_field_base_dipole
    kernel.restype = None
    kernel.argtypes = [
        ndpointer(np.float32, ndim=1, flags="C"),
        ndpointer(np.float32, ndim=1, flags="C"),
        ctypes.c_int,
        ndpointer(np.float32, ndim=2, flags="C"),
        ndpointer(np.float32, ndim=2, flags="C"),
    ]
    field = np.zeros(3, dtype=np.float32)
    kernel(
        field,
        np.asarray(point, dtype=np.float32),
        len(x),
        np.asarray(x, dtype=np.float32),
        np.asarray(mu, dtype=np.float32),
    )
    return field


def potential(point, x, mu):
    """Potential of point dipoles, k * mu . R / |R|^3 summed over the dipoles."""
    R = point - x
    r = np.linalg.norm(R, axis=1)
    return K * np.sum(np.einsum("ij,ij->i", mu, R) / r ** 3)


def test_field_on_the_axis_of_a_dipole():
    """On the axis of a dipole of moment m, at distance d, the field is 2 k m / d^3."""
    x = np.zeros((1, 3))
    mu = np.array([[0.0, 0.0, 0.5]])

    field = kernel_field([0.0, 0.0, 2.0], x, mu)

    np.testing.assert_allclose(field, [0.0, 0.0, 2 * K * 0.5 / 2.0 ** 3], rtol=1e-5, atol=1e-6)


def test_field_on_the_equator_of_a_dipole():
    """Perpendicular to the dipole, at distance d, the field is -k m / d^3 (opposite to it)."""
    x = np.zeros((1, 3))
    mu = np.array([[0.0, 0.0, 0.5]])

    field = kernel_field([2.0, 0.0, 0.0], x, mu)

    np.testing.assert_allclose(field, [0.0, 0.0, -K * 0.5 / 2.0 ** 3], rtol=1e-5, atol=1e-6)


def test_field_is_minus_the_gradient_of_the_potential():
    """Many dipoles with random positions and orientations."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-6.0, 6.0, (30, 3))
    mu = rng.normal(0.0, 0.5, (30, 3))
    point = rng.uniform(-1.0, 1.0, 3)

    step = 1e-5
    gradient = np.array(
        [
            (potential(point + step * e, x, mu) - potential(point - step * e, x, mu))
            / (2 * step)
            for e in np.eye(3)
        ]
    )

    np.testing.assert_allclose(kernel_field(point, x, mu), -gradient, rtol=1e-3, atol=1e-4)
