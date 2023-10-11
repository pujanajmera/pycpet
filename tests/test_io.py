from CPET.utils.parser import parse_pqr

import numpy as np


def test_pqr_parser():
    """Test the pqr parser"""
    test_pqr_file = "./test_files/test_large.pqr"
    x, Q = parse_pqr(test_pqr_file)
    x_test_case = np.array([x[10], x[16492], x[-10]])
    x_ground_truth = np.array(
        [
            [76.062, 94.681, 129.367],
            [114.752, 113.583, 105.331],
            [110.242, 117.554, 92.352],
        ]
    )

    assert np.allclose(
        x_test_case, x_ground_truth
    ), "Coordinates are not correctly parsed"
    assert Q[-10][0] == 0.068, "Charge is not correct - hetam+large atom count"
    assert Q[10][0] == -0.243, "Charge is not correct - small res number"
    assert Q[16492][0] == -0.101, "Charge is not correct - large res number"
    assert np.max(Q) < 1.0, "Charges are too large"
    assert np.min(Q) > -1.0, "Charges are too small"
    assert Q.shape[0] == x.shape[0], "Charges and coordinates are not the same length"


test_pqr_parser()
