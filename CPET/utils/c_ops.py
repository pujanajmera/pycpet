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
        self.array_3d_float = npct.ndpointer(dtype=np.double, ndim=3, flags="C")
        

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
            ctypes.c_int,
            ctypes.c_int,
            self.array_2d_float,
            self.array_1d_float,
        ]
        self.math.einsum_ij_i_batch.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            self.array_3d_float,
            self.array_2d_float,
        ]

        self.math.einsum_operation_batch.restype = None
        self.math.einsum_operation_batch.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            self.array_2d_float,
            self.array_1d_float,
            self.array_3d_float, 
            self.array_2d_float,
        ]

        #batch_size, len(Q), np.array(r_mag), np.array(Q), np.array(R), res


        self.math.einsum_operation.restype = None
        self.math.einsum_operation.argtypes = [
            ctypes.c_int,
            self.array_1d_float,
            self.array_1d_float,
            self.array_2d_float, 
            self.array_1d_float,
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
        res = np.zeros((A.shape[0]), dtype="float64")
        self.math.einsum_ij_i.restype = None
        self.math.einsum_ij_i(A.shape[0], A.shape[1], A, res)
        return res


    def einsum_ij_i_batch(self, A):
        # simply two vectors
        res = np.zeros((len(A), A[0].shape[0]), dtype="float64")
        self.math.einsum_ij_i_batch.restype = None
        self.math.einsum_ij_i_batch(len(A), A[0].shape[0], A[0].shape[1], A, res)
        res = res.reshape(res.shape[1], res.shape[0])
        return res


    def einsum_operation(self, R, r_mag, Q):
        res = np.zeros(3, dtype="float64")
        r_mag = r_mag.reshape(-1)
        R = R.reshape(r_mag.shape[0], 3)
        Q = Q.reshape(-1)
        #len_Q = len(Q)
        #print(r_mag.shape)
        #print(Q.shape)
        #print(R.shape)
        #print(res.shape)
        self.math.einsum_operation.restype = None
        self.math.einsum_operation(len(Q), np.array(r_mag), np.array(Q), np.array(R), res) 
        return res


    def einsum_operation_batch(self, R, r_mag, Q, batch_size):
        # res = np.ascontiguousarray(np.zeros(3, dtype="float64"))
        #print("einsum in")
        res = np.zeros((batch_size, 3), dtype="float64")
        self.math.einsum_operation_batch.restype = None
        Q = Q.reshape(-1)
        self.math.einsum_operation_batch(batch_size, len(Q), np.array(r_mag), np.array(Q), np.array(R), res)       
        return res
