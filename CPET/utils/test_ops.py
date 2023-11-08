from c_ops import Math_ops
import numpy as np
#from scipy.sparse import csr_matrix
import time

Math = Math_ops()

def test_ij_i(n_size=10000):
    random = np.random.rand(n_size, 3)
    time_start = time.time()
    einsum_random_np = np.einsum("ij->i", random).reshape(-1, 1)
    time_end = time.time()
    time_np = time_end - time_start
    time_start = time.time()
    einsum_random_c = Math.einsum_ij_i(random)
    time_end = time.time()
    time_c = time_end - time_start
    print("ratio us vs. np, einsum ij->i: ", time_c / time_np)
    assert np.allclose(
        einsum_random_np, einsum_random_c, rtol=1e-5
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(einsum_random_np - einsum_random_c)),
        np.max(einsum_random_np),
        np.max(einsum_random_c),
    )
    return 

def test_vec_addn(n_size=10000):
    D = np.random.rand(n_size)
    E = np.random.rand(n_size)
    time_start = time.time()
    tmp3 = Math.vecaddn(D, E)
    time_end = time.time()
    time_c = time_end - time_start
    time_start = time.time()
    sum_alt = D + E
    time_end = time.time()
    time_np = time_end - time_start
    assert np.allclose(
        D + E, tmp3
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(D + E - tmp3)), np.max(D + E), np.max(tmp3)
    )
    print("ratio us vs. np, vecaddn: ", time_c / time_np)
    return 


def test_einsum_operation(n_size=10000):
    rand_1 = np.random.rand(n_size, 3)
    rand_2 = np.random.rand(n_size, 1)
    rand_3 = np.random.rand(n_size, 1)

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
    print("ratio us vs. np ij,ij->j: ", c_time / np_time)
    assert np.allclose(
        res, res_test, rtol=1e-5
    ), "largest difference: {} where np has {} and c has {}".format(
        np.max(np.abs(res - res_test)), np.max(res), np.max(res_test)
    )

    return

if __name__ == "__main__":
    #ones = np.ones((1000000, 1), dtype="float64")
    # copy A_np along the dim 1 for 3 times
    n_size = 100000
    test_ij_i(n_size)
    test_vec_addn(n_size)
    test_einsum_operation(n_size)
    
    
