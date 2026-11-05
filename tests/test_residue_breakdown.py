import json
import shutil
import sys

import numpy as np
import pytest

from CPET.source.calculator import calculator
from CPET.source.scripts import residue_breakdown_analysis as rba

OPTIONS = {
    "CPET_method": "topo",
    "center": {"method": "first", "atoms": [{"CD": 2}]},
    "x": {"method": "mean", "atoms": [{"CG": 1}, {"CB": 1}]},
    "y": {"method": "mean", "atoms": [{"CA": 3}, {"CB": 3}]},
    # Remove the residue at the center, so no charge sits on a start point
    "filter": {"resid": ["GLN"], "resnum": [2]},
    "filter_intersect": True,
    "filter_radius": 6.0,
    "n_samples": 8,
    "dimensions": [1.0, 1.0, 1.0],
    "step_size": 0.05,
    "concur_slip": 2,
    "initializer": "uniform",
    "max_streamline_init": "fixed_rand",
    "dtype": "float32",
}


def test_get_residues_lists_each_residue_once():
    """Atoms of the same residue are next to each other. This used to raise an error."""
    resnames = ["SER", "SER", "SER", "ALA", "ALA", "SER"]
    resids = ["1", "1", "1", "2", "2", "3"]

    assert rba.get_residues(resnames, resids) == [
        {"resname": "SER", "resid": "1"},
        {"resname": "ALA", "resid": "2"},
        {"resname": "SER", "resid": "3"},
    ]


def test_get_residues_skips_water():
    resnames = ["SER", "WAT", "WAT", "ALA"]
    resids = ["1", "2", "2", "3"]

    assert [r["resname"] for r in rba.get_residues(resnames, resids)] == ["SER", "ALA"]


def test_get_residues_rejects_a_repeated_residue():
    """The same residue coming back after another one, for example in another chain."""
    with pytest.raises(ValueError, match="repeated"):
        rba.get_residues(["SER", "ALA", "SER"], ["1", "2", "1"])


def test_load_hist_accepts_any_whitespace(tmp_path):
    top = tmp_path / "a.top"
    top.write_text("0.5 1.0\n0.6\t1.2\n  0.7   0.9  \n")

    hist = rba.load_hist("a.pdb", str(tmp_path))

    np.testing.assert_allclose(hist, [[0.5, 1.0], [0.6, 1.2], [0.7, 0.9]])


@pytest.fixture
def run_script(tmp_path, test_files, monkeypatch):
    """Set up a folder of one pdb and its topology, and a function that runs the script."""
    pdbs = tmp_path / "pdbs"
    tops = tmp_path / "tops"
    pdbs.mkdir()
    tops.mkdir()
    shutil.copy(test_files / "test_large.pdb", pdbs / "frame1.pdb")

    # The full topology comes from the same system and options
    system = calculator(OPTIONS, path_to_pdb=str(pdbs / "frame1.pdb"))
    np.savetxt(tops / "frame1.top", system.compute_topo_complete_c_shared())

    (tmp_path / "options.json").write_text(json.dumps(OPTIONS))
    monkeypatch.chdir(tmp_path)

    def run():
        argv = ["residue_breakdown_analysis.py", "-t", str(tops), "-p", str(pdbs)]
        argv += ["-c", "0,3", "-cx", "0.5", "-d", "0,3", "-dx", "0.5"]
        argv += ["-o", str(tmp_path / "options.json")]
        monkeypatch.setattr(sys, "argv", argv)
        rba.main()

    return run, system, pdbs, tops, tmp_path


def test_script_writes_the_residue_breakdown(run_script):
    run, system, _, _, folder = run_script

    run()

    # One entry per residue within the radius, none for water
    residues = {(atom[2], atom[3]) for atom in system.ID if atom[2] != "WAT"}
    listed = (folder / "residue_list.txt").read_text().splitlines()
    averages = (folder / "chi_squared_avg.txt").read_text().splitlines()
    assert len(residues) > 1
    assert len(listed) == len(residues)
    assert len(averages) == len(residues)

    # Sorted from the smallest mean distance, each between 0 and 1 for these histograms
    means = [eval(line)[1] for line in averages]
    assert means == sorted(means)
    assert all(0.0 <= m <= 1.0 for m in means)


def test_script_needs_a_pdb_for_every_topology(run_script):
    run, _, pdbs, tops, _ = run_script
    shutil.copy(tops / "frame1.top", tops / "frame2.top")

    with pytest.raises(ValueError, match="frame2.top does not have a corresponding pdb"):
        run()


def test_script_needs_a_topology_for_every_pdb(run_script):
    run, _, pdbs, tops, _ = run_script
    shutil.copy(pdbs / "frame1.pdb", pdbs / "frame2.pdb")

    with pytest.raises(ValueError, match="frame2.pdb does not have a corresponding topology"):
        run()


def test_script_ignores_other_files_in_the_folders(run_script):
    run, _, pdbs, tops, folder = run_script
    (tops / "notes.txt").write_text("not a topology")
    (pdbs / "README").write_text("not a pdb")

    run()

    assert (folder / "chi_squared_avg.txt").exists()
