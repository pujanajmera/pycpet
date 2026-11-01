import os
from pathlib import Path

# topo gives incorrect results with more than one OpenMP thread, so tests run on one.
# This has to be set before the C extension is loaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

TESTS_DIR = Path(__file__).parent
REPO_DIR = TESTS_DIR.parent


@pytest.fixture(scope="session")
def test_files():
    """Folder with the small PDB/PQR files used by the tests."""
    return TESTS_DIR / "test_files"


@pytest.fixture
def synthetic_data():
    """Four charges: two in residue WAT 10, then FE 651 and NI 652."""
    x = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.1, 1.0, 0.0], [5.0, 5.0, 5.0]]
    )
    Q = np.array([[1.0], [-0.5], [0.3], [1.2]])
    # (atom_number, atom_type, resid, resnum, chain), as built by the calculator
    ID = np.array(
        [
            (1001, "O", "WAT", 10, "A"),
            (1002, "H", "WAT", 10, "A"),
            (2001, "FE", "FE", 651, "A"),
            (3001, "NI", "NI", 652, "A"),
        ],
        dtype=object,
    )
    return {"x": x, "Q": Q, "ID": ID}
