import numpy as np
from subprocess import Popen, PIPE
import ctypes
import numpy.ctypeslib as npct


class Math_ops:
    def __init__(self, shared_loc=None):
        if shared_loc is None:
            self.math = ctypes.CDLL("./math_module.so")
        else:
            self.math = ctypes.CDLL(shared_loc)

        # creates pointers to array data types
        self.array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags="C")
        self.array_1d_float = npct.ndpointer(dtype=np.double, ndim=1, flags="C")
        self.array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C")
        self.array_2d_float = npct.ndpointer(dtype=np.double, ndim=2, flags="C")

        # initial arguement
        self.math.sparse_dot.argtypes = [
            self.array_1d_float,
            self.array_1d_int,
            ctypes.c_int,
            self.array_1d_int,
            ctypes.c_int,
            self.array_1d_float,
            ctypes.c_int,
            self.array_1d_float,
            ctypes.c_int,
        ]
        self.math.vecaddn.argtypes = [
            self.array_1d_float,
            self.array_1d_float,
            self.array_1d_float,
            ctypes.c_int,
        ]

        self.math.dot.argtypes = [
            self.array_1d_float,
            self.array_2d_float,
            self.array_1d_float,
            ctypes.c_int,
            ctypes.c_int,
        ]

        self.math.einsum_ij_i.argtypes = [
            self.array_1d_float,
            self.array_2d_float,
            ctypes.c_int,
            ctypes.c_int,
        ]

        #self.math.einsum_ij_ij_to_ij_i.argtypes = [
        #    self.array_1d_float,
        #    self.array_2d_float,
        #    self.array_2d_float,
        #    self.array_2d_float,
        #    ctypes.c_int,
        #]

        self.math.einsum_operation.restype = None
        self.math.einsum_operation.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        # self.math.test_trans.argtypes = None

    def sparse_dot(self, A, B):
        # b is just a single vector, not a sparse matrix
        # a is a full sparse matrix
        res = np.zeros(len(A.data), dtype="float64")
        self.math.sparse_dot.restype = None

        self.math.sparse_dot(
            res,
            A.indptr,
            len(A.indptr),
            A.indices,
            len(A.indices),
            A.data,
            len(A.data),
            B.astype("double"),
            len(B),
        )
        return res

    def dot(self, A, B):
        # b is just a single vector, not a sparse matrix
        # a is a full sparse matrix
        res = np.zeros(len(B.data), dtype="float64")
        self.math.dot.restype = None

        self.math.dot(
            res,
            A.astype("double"),
            B.astype("double"),
            len(A),
            len(B),
        )
        return res

    def vecaddn(self, A, B):
        # simply two vectors
        res = np.zeros(len(A), dtype="float64")
        self.math.vecaddn.restype = None
        # self.math.vecaddn.restype = npct.ndpointer(
        #    dtype=self.array_1d_float, shape=len(A)
        # )
        self.math.vecaddn(res, A, B, len(A))
        return res

    def einsum_ij_i(self, A):
        # simply two vectors
        res = np.zeros(A.shape[0], dtype="float64")
        # self.math.einsum_ij_i.restype = npct.ndpointer(
        #    dtype=self.array_1d_float, shape=int(A.shape[0])
        # )
        # print(A.shape[0], A.shape[1])
        self.math.einsum_ij_i.restype = None
        self.math.einsum_ij_i(res, A, int(A.shape[0]), int(A.shape[1]))
        return res.reshape(-1, 1)

    def einsum_operation(self, R, r_mag, Q):
        # res = np.ascontiguousarray(np.zeros(3, dtype="float64"))
        res = np.zeros(3, dtype="float64")
        # R = np.ascontiguousarray(R, dtype=np.float64)
        # r_mag = np.ascontiguousarray(r_mag, dtype=np.float64)
        # flatten R, r_mag, Q
        # R = np.ascontiguousarray(R.flatten(), dtype=np.float64)
        # r_mag = np.ascontiguousarray(r_mag.flatten(), dtype=np.float64)
        R = R.reshape(-1)
        r_mag = r_mag.reshape(-1)
        Q = Q.reshape(-1)
        len_Q = len(Q)
        self.math.einsum_operation.restype = None
        self.math.einsum_operation(R, r_mag, Q, len_Q, res)
        # print(res)
        # print(res.shape)
        return res
