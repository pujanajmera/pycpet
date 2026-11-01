import numpy as np
import pytest

from CPET.source.calculator import calculator
from CPET.utils.calculator import Inside_Box, compute_curv_and_dist

K = 14.3996451  # V * Angstrom / e

OPTIONS = {
    "CPET_method": "topo",
    "center": {"method": "first", "atoms": [{"CD": 2}]},
    "x": {"method": "mean", "atoms": [{"CG": 1}, {"CB": 1}]},
    "y": {"method": "mean", "atoms": [{"CA": 3}, {"CB": 3}]},
    # Remove the residue at the center, so no charge sits on a start point
    "filter": {"resid": ["GLN"], "resnum": [2]},
    "filter_intersect": True,
    "filter_radius": 15.0,
    "n_samples": 27,
    "dimensions": [1.0, 1.0, 1.0],
    "step_size": 0.05,
    "concur_slip": 2,
    "initializer": "uniform",
    "max_streamline_init": "fixed_rand",
    "dtype": "float32",
}


@pytest.fixture(scope="module")
def topo(test_files):
    return calculator(OPTIONS, path_to_pdb=str(test_files / "test_large.pdb"))


def reference_topology(topo):
    """Follow each streamline with plain NumPy in double precision.

    Takes the same steps as the C code: move step_size along the field direction,
    stop on leaving the box or after the streamline's random maximum number of steps,
    then take the distance and curvature from the first and last three points.
    """
    x = topo.x.astype(np.float64)
    Q = topo.Q.astype(np.float64)

    def step(p):
        R = p - x
        E = K * (Q * R / np.linalg.norm(R, axis=1, keepdims=True) ** 3).sum(axis=0)
        return p + topo.step_size * E / np.linalg.norm(E)

    rows = []
    for start, n_steps in zip(topo.random_start_points, topo.random_max_samples):
        end = start.astype(np.float64)
        for _ in range(n_steps):
            end = step(end)
            if not Inside_Box(end, topo.dimensions):
                break
        start_1 = step(start)
        start_2 = step(start_1)
        end_1 = step(end)
        end_2 = step(end_1)
        rows.append(compute_curv_and_dist(start, start_1, start_2, end, end_1, end_2)[:2])
    return np.array(rows)


def test_start_points_fill_the_box(topo):
    """27 samples make a 3 x 3 x 3 grid of start points inside the box."""
    assert topo.n_samples == 27
    assert topo.random_start_points.shape == (27, 3)
    assert (np.abs(topo.random_start_points) < topo.dimensions).all()
    assert topo.random_max_samples.shape == (27,)
    assert (topo.random_max_samples >= 1).all()


def test_n_samples_rounds_up_to_a_cube(test_files):
    options = dict(OPTIONS, n_samples=30)

    t = calculator(options, path_to_pdb=str(test_files / "test_large.pdb"))

    assert t.n_samples == 64
    assert t.random_start_points.shape == (64, 3)


def test_c_topology_matches_reference(topo):
    """Distance and curvature of every streamline agree with the NumPy reference."""
    hist = topo.compute_topo_complete_c_shared()

    assert hist.shape == (27, 2)
    assert np.isfinite(hist).all()
    np.testing.assert_allclose(hist, reference_topology(topo), rtol=1e-3, atol=1e-3)


def test_single_process_topology_matches_c(topo):
    """compute_topo_single, the plain Python version, agrees with the C extension."""
    hist = topo.compute_topo_single()

    assert hist.shape == (27, 2)
    assert np.isfinite(hist).all()
    np.testing.assert_allclose(
        hist, topo.compute_topo_complete_c_shared(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(hist, reference_topology(topo), rtol=1e-3, atol=1e-3)


def test_c_topology_is_reproducible(topo):
    """With fixed_rand, two runs give the same result."""
    np.testing.assert_array_equal(
        topo.compute_topo_complete_c_shared(), topo.compute_topo_complete_c_shared()
    )


@pytest.mark.gpu
def test_gpu_topology_matches_c(topo):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    gpu = topo.compute_topo_GPU_batch_filter()
    cpu = topo.compute_topo_complete_c_shared()

    def by_distance(h):
        return h[np.lexsort((np.round(h[:, 1], 3), np.round(h[:, 0], 3)))]

    np.testing.assert_allclose(by_distance(gpu), by_distance(cpu), rtol=1e-2, atol=1e-2)


def test_filter_in_box_removes_charges_in_the_box(test_files):
    """filter_in_box removes the charges inside the box, here just the center atom."""
    options = dict(OPTIONS)
    del options["filter"], options["filter_intersect"]
    pdb = str(test_files / "test_large.pdb")

    unfiltered = calculator(dict(options, filter_in_box=False), path_to_pdb=pdb)
    filtered = calculator(dict(options, filter_in_box=True), path_to_pdb=pdb)

    def n_inside(t):
        return (np.abs(t.x) < np.array(t.dimensions)).all(axis=1).sum()

    assert n_inside(unfiltered) == 1
    assert n_inside(filtered) == 0
    assert len(filtered.x) == len(unfiltered.x) - 1
    assert len(filtered.Q) == len(filtered.ID) == len(filtered.x)
