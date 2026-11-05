import os
import time

import numpy as np
import pytest

import CPET.source.cluster as cluster_module
from CPET.source.cluster import cluster

HEADER = (
    "#Sample Density: 3 3 3; Volume: Box: 1 1 1\n#Frame 0\n#Center: 0 0 0\n"
    "#Basis Matrix:\n# 1 0 0\n# 0 1 0\n# 0 0 1\n"
)
GRID = np.array(np.meshgrid(*[np.linspace(-1, 1, 3)] * 3, indexing="ij")).reshape(3, -1).T


def write_files(folder, kind, scale=1.0, n_files=9):
    """Write potential (_esp.dat) or field (_efield.dat) files, in three groups."""
    folder.mkdir()
    rng = np.random.default_rng(0)
    n_values = 1 if kind == "esp" else 3
    for i in range(n_files):
        values = rng.normal(0.0, 0.05, (len(GRID), n_values)) + (i % 3 + 1) * scale
        suffix = "_esp.dat" if kind == "esp" else "_efield.dat"
        with open(folder / f"s{i}{suffix}", "w") as f:
            f.write(HEADER)
            np.savetxt(f, np.c_[GRID, values], fmt="%.4f")


def run(method, inputpath, outputpath):
    options = {
        "CPET_method": method,
        "inputpath": str(inputpath),
        "outputpath": str(outputpath),
        "rank": 2,
    }
    return cluster(options)


@pytest.fixture
def folders(tmp_path):
    write_files(tmp_path / "esp", "esp")
    write_files(tmp_path / "esp_other", "esp", scale=5.0)
    write_files(tmp_path / "fields", "field")
    out = tmp_path / "out"
    out.mkdir()
    return tmp_path / "esp", tmp_path / "esp_other", tmp_path / "fields", out


@pytest.fixture
def tensor_builds(monkeypatch):
    """Count how many times the 5D tensor is built from the files."""
    builds = []
    real = cluster_module.make_5d_tensor

    def counting(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cluster_module, "make_5d_tensor", counting)
    return builds


def test_tensor_is_reused_for_the_same_files(folders, tensor_builds):
    esp, _, _, out = folders

    run("cluster_volume_esp_tensor", esp, out)
    run("cluster_volume_esp_tensor", esp, out)

    assert len(tensor_builds) == 1


def test_field_tensor_is_not_used_for_potentials(folders):
    """A leftover field tensor has 3 values per grid point and used to be loaded."""
    esp, _, fields, out = folders
    run("cluster_volume_tensor", fields, out)
    assert np.load(out / "5d_tensor.npy").shape == (9, 3, 3, 3, 3)

    clustering = run("cluster_volume_esp_tensor", esp, out)

    assert clustering.full_tensor.shape == (9, 3, 3, 3, 1)
    assert np.load(out / "5d_tensor.npy").shape == (9, 3, 3, 3, 1)


def test_tensor_of_other_files_is_not_used(folders):
    """Same number of files, but newer ones with different values."""
    esp, esp_other, _, out = folders
    first = run("cluster_volume_esp_tensor", esp, out)

    # The other files are written after the saved tensor
    later = time.time() + 10
    for f in esp_other.iterdir():
        os.utime(f, (later, later))
    second = run("cluster_volume_esp_tensor", esp_other, out)

    assert second.full_tensor.max() > 4 * first.full_tensor.max()


def test_tensor_for_a_different_number_of_files_is_not_used(folders, tensor_builds):
    esp, _, _, out = folders
    run("cluster_volume_esp_tensor", esp, out)

    next(esp.iterdir()).unlink()
    clustering = run("cluster_volume_esp_tensor", esp, out)

    assert len(tensor_builds) == 2
    assert clustering.full_tensor.shape[0] == 8


def test_unreadable_tensor_is_built_again(folders, tensor_builds):
    esp, _, _, out = folders
    (out / "5d_tensor.npy").write_text("not a tensor")
    later = time.time() + 10
    os.utime(out / "5d_tensor.npy", (later, later))

    clustering = run("cluster_volume_esp_tensor", esp, out)

    assert len(tensor_builds) == 1
    assert clustering.full_tensor.shape == (9, 3, 3, 3, 1)
