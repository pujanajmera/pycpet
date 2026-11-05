import sys

import numpy as np
import pytest

from CPET.source.scripts import field_topology_dipole as dipole_script

# index, x, y, z (Angstrom), then the dipole moment in a.u., clear of the default box
DIPOLES = (
    "index x y z mx my mz\n"
    "0 3.0 0.5 0.2 0.5 0.1 0.0\n"
    "1 -2.5 1.0 -0.4 0.0 0.3 0.2\n"
    "2 0.3 -3.0 1.0 0.1 0.0 -0.4\n"
)


@pytest.fixture
def run_script(tmp_path, monkeypatch):
    """Run the script on one dipole file with the given flags, and return its topology."""
    dipole_dir = tmp_path / "dipoles"
    dipole_dir.mkdir()
    (dipole_dir / "frame1.dip").write_text(DIPOLES)
    monkeypatch.chdir(tmp_path)

    def run(*flags):
        # The maximum number of steps of each streamline is random, so seed it
        np.random.seed(0)
        argv = ["field_topology_dipole.py", "-d", str(dipole_dir), "-n", "8", *flags]
        monkeypatch.setattr(sys, "argv", argv)
        dipole_script.main()
        return np.loadtxt(tmp_path / "frame1.top")

    return run


def test_script_runs_with_the_default_frame_and_box(run_script):
    """The default -y used to equal -x, and the default -dim made the C call fail."""
    topology = run_script()

    assert topology.shape == (8, 2)
    assert np.isfinite(topology).all()


def test_defaults_are_the_standard_frame_and_a_unit_box(run_script):
    default = run_script()

    explicit = run_script(
        "-c", "0", "0", "0", "-x", "1", "0", "0", "-y", "0", "1", "0", "-dim", "1", "1", "1"
    )

    np.testing.assert_array_equal(default, explicit)
