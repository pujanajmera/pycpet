import pytest

from CPET.source.calculator import calculator

OPTIONS = {
    "CPET_method": "topo",
    "center": {"method": "first", "atoms": [{"CD": 2}]},
    "x": {"method": "mean", "atoms": [{"CG": 1}, {"CB": 1}]},
    "y": {"method": "mean", "atoms": [{"CA": 3}, {"CB": 3}]},
    "dimensions": [1.0, 1.0, 1.0],
    "step_size": 0.05,
    "n_samples": 8,
    "concur_slip": 1,
    "max_streamline_init": "fixed_rand",
    "dtype": "float32",
}

FILTERS = {
    "radius": {"filter_radius": 10.0},
    "atomspec": {"filter": {"resid": ["GLN"]}, "filter_intersect": True},
    "atomspec and radius": {
        "filter": {"resid": ["GLN"], "resnum": [2]},
        "filter_intersect": False,
        "filter_radius": 10.0,
    },
    "radius and box": {"filter_radius": 10.0, "filter_in_box": True},
}


def identity(atom):
    return tuple(str(value) for value in atom)


@pytest.fixture(scope="module")
def unfiltered(test_files):
    return calculator(dict(OPTIONS), path_to_pdb=str(test_files / "test_large.pdb"))


def test_no_filter_removes_nothing(unfiltered):
    assert unfiltered.filter_idxs == []


@pytest.mark.parametrize("extra", FILTERS.values(), ids=FILTERS.keys())
def test_filter_idxs_are_the_original_positions_of_the_removed_charges(
    test_files, unfiltered, extra
):
    calc = calculator(dict(OPTIONS, **extra), path_to_pdb=str(test_files / "test_large.pdb"))

    assert len(calc.filter_idxs) == 1
    removed = set(calc.filter_idxs[0].tolist())
    kept = [i for i in range(len(unfiltered.ID)) if i not in removed]

    assert len(kept) == len(calc.x)
    assert [identity(unfiltered.ID[i]) for i in kept] == [identity(a) for a in calc.ID]
