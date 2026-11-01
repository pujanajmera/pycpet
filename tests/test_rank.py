import logging

import numpy as np
import pytest

import CPET.utils.calculator as calc
from CPET.utils.calculator import determine_rank

# Reconstruction errors for ranks 1 to 6; the elbow is at rank 2
ERRORS = [0.1, 0.01, 0.001, 0.0005, 0.0004, 0.00035]


@pytest.fixture
def rank_with_errors(monkeypatch):
    """determine_rank with the CP decomposition replaced by a table of errors."""

    def run(errors, threshold, min_rank=1):
        ranks = list(range(min_rank, min_rank + len(errors)))
        table = dict(zip(ranks, errors))
        monkeypatch.setattr(calc, "reduce_tensor", lambda tensor, rank: (None, table[rank]))
        return determine_rank(None, threshold, ranks[-1], min_rank)

    return run


def test_elbow_rank_is_used_when_its_error_is_low_enough(rank_with_errors):
    assert rank_with_errors(ERRORS, threshold=0.01) == 2


def test_lowest_rank_below_threshold_when_the_elbow_is_not_good_enough(rank_with_errors):
    # The elbow (rank 2) has error 0.01; rank 5 is the first below 0.00045
    assert rank_with_errors(ERRORS, threshold=0.00045) == 5


def test_lowest_error_rank_when_no_rank_reaches_the_threshold(rank_with_errors, caplog):
    """This used to return rank 50, although only ranks 1 to 6 were tried."""
    with caplog.at_level(logging.WARNING):
        rank = rank_with_errors(ERRORS, threshold=1e-9)

    assert rank == 6
    assert "No rank from 1 to 6" in caplog.text


def test_min_rank_above_one(rank_with_errors):
    """Errors are for ranks 3 to 8; the elbow is rank 4, whose error is 0.01.

    That is above the threshold, so the first rank below it, 5, is used. The error
    used to be read from the wrong place in the list, which gave rank 4.
    """
    assert rank_with_errors(ERRORS, threshold=0.005, min_rank=3) == 5


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # kneed divides by zero on a flat curve
def test_no_elbow_to_find(rank_with_errors):
    """Two ranks, or a flat curve, have no elbow. This used to raise a TypeError."""
    assert rank_with_errors([0.01, 0.0001], threshold=0.001) == 2
    assert rank_with_errors([0.003] * 5, threshold=0.001) == 1


def test_rank_of_a_real_tensor():
    """A tensor that is exactly rank 2 is described well by a rank of at least 2."""
    rng = np.random.default_rng(0)
    factors = [rng.normal(size=(n, 2)) for n in (6, 4, 4, 4, 3)]
    tensor = np.einsum("ar,br,cr,dr,er->abcde", *factors)

    rank = determine_rank(tensor, threshold=1e-3, max_rank=4)

    assert 2 <= rank <= 4
