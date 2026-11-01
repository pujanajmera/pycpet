"""
Streamlines should not depend on the number of OpenMP threads.

The C extension reads OMP_NUM_THREADS once, when it is loaded, so each thread count is
run in its own Python process.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

CHARGE_STREAMLINES = textwrap.dedent(
    """
    import sys
    import numpy as np
    from CPET.utils.calculator import calculate_thread_c_shared

    rng = np.random.default_rng(0)
    x = rng.uniform(-8, 8, (6000, 3))
    x = x[np.linalg.norm(x, axis=1) > 2.0].astype(np.float32)
    Q = rng.uniform(-1, 1, (len(x), 1)).astype(np.float32)
    starts = rng.uniform(-0.8, 0.8, (32, 3)).astype(np.float32)
    box = np.array([1, 1, 1], dtype=np.float32)

    result = [
        calculate_thread_c_shared(x_0=s, n_iter=5, x=x, Q=Q, step_size=0.05, dimensions=box)
        for s in starts
    ]
    np.save(sys.argv[1], np.array(result))
    """
)


def run_with_threads(script, n_threads, output):
    """Run a script in a new process with OMP_NUM_THREADS set, and load what it saved."""
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    subprocess.run([sys.executable, "-c", script, str(output)], check=True, env=env)
    return np.load(output)


def test_charge_streamlines_do_not_depend_on_the_thread_count(tmp_path):
    """With shared scratch variables, threads used to overwrite each other's values."""
    one = run_with_threads(CHARGE_STREAMLINES, 1, tmp_path / "one.npy")
    four = run_with_threads(CHARGE_STREAMLINES, 4, tmp_path / "four.npy")

    assert one.shape == (32, 2)
    assert np.isfinite(one).all()
    np.testing.assert_allclose(four, one, rtol=1e-4, atol=1e-5)
