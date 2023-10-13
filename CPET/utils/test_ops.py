from c_ops import Math_ops
import numpy as np
from scipy.sparse import csr_matrix
import time

if __name__ == "__main__":
    A_np = np.array([[4, 0, 6], [0, 0, 3], [4, 0, 6], [0, 8, 9]], dtype="float64").T
    ones = np.ones((10000, 1), dtype="float64")
    # copy A_np along the dim 1 for 3 times
    random = np.random.rand(1000, 3)
    B = np.array([3, 4, 5], dtype="float64")

    # C = np.array([[4, 5, 6], [6, 7, 6]], dtype="float64")
    D = np.array([1, 1, 1, 1, 1, 1], dtype="float64")
    E = np.array([4, 6, 4, 3, 1, 3], dtype="float64")
    Math = Math_ops()

    B = np.array([1.0, 2.0, 3.0])
    A = csr_matrix(A_np)
    # print("number of nonzero values in each row:")
    # print("a indptr: ", A.indptr)
    # print("length of data: ", int(len(A.data)))
    # print(A_np.shape)

    einsum_py = np.einsum("ij->i", ones).reshape(-1, 1)
    einsum_c = Math.einsum_ij_i(ones)
    """
    assert np.allclose(
        einsum_py, einsum_c
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(einsum_py - einsum_c)), np.max(einsum_py), np.max(einsum_c)
    )
    """
    time_start = time.time()
    einsum_random_np = np.einsum("ij->i", random).reshape(-1, 1)
    time_end = time.time()
    time_np = time_end - time_start
    # print("time taken np: ", time_end - time_start)
    time_start = time.time()
    einsum_random_c = Math.einsum_ij_i(random)
    time_end = time.time()
    time_c = time_end - time_start
    print(time_c / time_np)
    # print("time taken c: ", time_end - time_start)
    # print(einsum_random_np.shape)
    # print(einsum_random_c.shape)
    assert np.allclose(
        einsum_py, einsum_c, rtol=1e-5
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(einsum_random_np - einsum_random_c)),
        np.max(einsum_random_np),
        np.max(einsum_random_c),
    )

    tmp3 = Math.vecaddn(D, E)
    assert np.allclose(
        D + E, tmp3
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(D + E - tmp3)), np.max(D + E), np.max(tmp3)
    )

    # make 1000, 3 matrix
    rand_1 = np.random.rand(30000, 3)
    rand_2 = np.random.rand(30000, 1)
    rand_3 = np.random.rand(30000, 1)

    time_start = time.time()
    res = np.einsum("ij,ij,ij->j", rand_1, 1 / rand_2, rand_3) * 14.3996451
    time_end = time.time()
    np_time = time_end - time_start
    # print("time taken np: ", np_time)
    time_start = time.time()
    res_test = Math.einsum_operation(rand_1, 1 / rand_2, rand_3)
    time_end = time.time()
    c_time = time_end - time_start
    # print("time taken c: ", c_time)
    print(c_time / np_time)
    assert np.allclose(
        res, res_test, rtol=1e-5
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(res - res_test)), np.max(res), np.max(res_test)
    )

    # print(tmp3)
    # print(tmp3[0].item())
    # Math.test_trans()
    # print(tmp1)
    # print(tmp_1_dense)
    # print(tmp2)
    # print(tmp3)
