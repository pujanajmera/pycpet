import os
import subprocess
import sys
import textwrap

import numpy as np

from CPET.utils.calculator import Math

K = 14.3996454784255  # V * Angstrom / e


def random_system(n, seed=0):
    """Charges, dipoles and traceless symmetric quadrupoles, as (xx, xy, xz, yy, yz, zz)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6.0, 6.0, (n, 3))
    Q = rng.uniform(-1.0, 1.0, n)
    mu = rng.normal(0.0, 0.5, (n, 3))
    A = rng.normal(0.0, 0.3, (n, 3, 3))
    T = (A + A.transpose(0, 2, 1)) / 2
    T -= np.eye(3) * np.trace(T, axis1=1, axis2=2)[:, None, None] / 3
    t = np.stack(
        [T[:, 0, 0], T[:, 0, 1], T[:, 0, 2], T[:, 1, 1], T[:, 1, 2], T[:, 2, 2]], axis=1
    )
    return x, Q, mu, T, t


def potential(point, x, Q, mu, T):
    """Potential of charges, dipoles and quadrupoles (Buckingham convention, e * A^2)."""
    R = point - x
    r = np.linalg.norm(R, axis=1)
    terms = (
        Q / r
        + np.einsum("ij,ij->i", mu, R) / r ** 3
        + np.einsum("ni,nij,nj->n", R, T, R) / r ** 5
    )
    return K * terms.sum()


def test_field_is_minus_the_gradient_of_the_potential():
    x, Q, mu, T, t = random_system(20)
    point = np.random.default_rng(1).uniform(-1.0, 1.0, 3)

    step = 1e-5
    gradient = np.array(
        [
            (potential(point + step * e, x, Q, mu, T) - potential(point - step * e, x, Q, mu, T))
            / (2 * step)
            for e in np.eye(3)
        ]
    )
    field = Math.compute_looped_field_amoeba(point[None], x, Q, mu, t)[0]

    np.testing.assert_allclose(field, -gradient, rtol=1e-6, atol=1e-8)


def test_without_dipoles_and_quadrupoles_it_is_the_point_charge_field():
    x, Q, _, _, _ = random_system(50)
    points = np.random.default_rng(2).uniform(-1.0, 1.0, (10, 3))

    amoeba = Math.compute_looped_field_amoeba(points, x, Q, np.zeros((50, 3)), np.zeros((50, 6)))
    charges = Math.compute_looped_field(points, x, Q)

    np.testing.assert_allclose(amoeba, charges, rtol=1e-4, atol=1e-4)


GRID_FIELD = textwrap.dedent(
    """
    import sys
    import numpy as np
    from CPET.utils.calculator import Math

    rng = np.random.default_rng(0)
    x = rng.uniform(-8, 8, (3000, 3))
    Q = rng.uniform(-1, 1, 3000)
    points = rng.uniform(-1, 1, (400, 3))
    np.save(sys.argv[1], Math.compute_looped_field(points, x, Q))
    """
)


def test_grid_field_does_not_depend_on_the_thread_count(tmp_path):
    """The grid kernel keeps its scratch arrays outside the loop, so it has to stay serial."""
    results = []
    for n_threads in (1, 4):
        output = tmp_path / f"{n_threads}.npy"
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        subprocess.run([sys.executable, "-c", GRID_FIELD, str(output)], check=True, env=env)
        results.append(np.load(output))

    np.testing.assert_allclose(results[1], results[0], rtol=1e-4, atol=1e-5)
