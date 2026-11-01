import numpy as np
import pytest

from CPET.utils.visualize import fac, visualize_field


def write_field_file(path, n_points, half_widths):
    """Write a field file on a regular grid, with a constant field plus noise."""
    axes = [np.linspace(-w, w, n) for n, w in zip(n_points, half_widths)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    field = np.random.default_rng(0).normal(0.0, 1.0, grid.shape) + [3.0, 0.0, 0.0]
    nx, ny, nz = n_points
    with open(path, "w") as f:
        f.write(
            f"#Sample Density: {nx} {ny} {nz}; "
            f"Volume: Box: {half_widths[0]} {half_widths[1]} {half_widths[2]}\n"
            "#Frame 0\n#Center: 0 0 0\n#Basis Matrix:\n# 1 0 0\n# 0 1 0\n# 0 0 1\n"
        )
        np.savetxt(f, np.c_[grid, field], fmt="%.4f")


def count_arrows(tmp_path, n_points, half_widths, sparsify_factor):
    """Run visualize_field and count the arrows in the .bild file."""
    field_file = tmp_path / "test_efield.dat"
    write_field_file(field_file, n_points, half_widths)

    options = {"visualization": {"cutoff": 0, "sparsify_factor": sparsify_factor}}
    visualize_field(str(field_file), str(tmp_path), options)

    bild = (tmp_path / "test_efield.bild").read_text().splitlines()
    return sum(line.startswith(".arrow") for line in bild)


def test_fac_picks_the_closest_factor():
    assert fac(12, 4) == 4  # 4 is a factor of 12
    assert fac(12, 5) == 6  # 4 and 6 are equally close, the larger wins
    assert fac(11, 2) == 1  # prime: only 1 and 11
    assert fac(21, 2) == 3  # 1 and 3 are equally close, the larger wins


def test_no_sparsification_draws_every_arrow_but_the_weakest(tmp_path):
    # The weakest arrow normalizes to 0, which is not above the 0th percentile cutoff
    assert count_arrows(tmp_path, (9, 9, 9), (0.4, 0.4, 0.4), 1) == 9 ** 3 - 1


def test_sparsification_thins_a_grid_with_a_common_factor(tmp_path):
    # 9 points per axis: factor 2 snaps to 3, leaving 3 x 3 x 3 arrows
    assert count_arrows(tmp_path, (9, 9, 9), (0.4, 0.4, 0.4), 2) == 27


def test_sparsification_has_no_effect_on_prime_grids(tmp_path):
    assert count_arrows(tmp_path, (11, 11, 11), (0.5, 0.5, 0.5), 2) == 11 ** 3 - 1


def test_sparsification_uses_every_axis(tmp_path):
    """x has 21 points (factor 3) but y and z have 11 (prime), so no common factor.

    The factor used to come from x alone, which left a single line of 7 arrows.
    """
    assert count_arrows(tmp_path, (21, 11, 11), (1.0, 0.5, 0.5), 2) == 21 * 11 * 11 - 1
