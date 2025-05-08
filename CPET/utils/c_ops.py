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
        self.array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags="C")
        self.array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags="C")
        self.array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags="C")
        self.array_3d_float = npct.ndpointer(dtype=np.float32, ndim=3, flags="C")

        self.array_3d_double = npct.ndpointer(dtype=np.double, ndim=3, flags="C")
        self.array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags="C")
        self.array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags="C")

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

        self.math.einsum_operation.restype = None
        self.math.einsum_operation.argtypes = [
            ctypes.c_int,
            self.array_1d_float,
            self.array_1d_float,
            self.array_2d_float,
            self.array_1d_float,
        ]

        self.math.thread_operation.restype = None
        self.math.thread_operation.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            self.array_1d_float,
            self.array_1d_float,
            self.array_2d_float,
            self.array_1d_float,
            self.array_1d_float,
        ]

        self.math.calc_field.restype = None
        self.math.calc_field.argtypes = [
            self.array_1d_float,
            self.array_1d_float,
            ctypes.c_int,
            self.array_2d_float,
            self.array_1d_float,
        ]

        self.math.calc_field_base.restype = None
        self.math.calc_field_base.argtypes = [
            self.array_1d_float,
            self.array_1d_float,
            ctypes.c_int,
            self.array_2d_float,
            self.array_1d_float,
        ]

        self.math.calc_esp_base.restype = None
        self.math.calc_esp_base.argtypes = [
            self.array_1d_float,
            self.array_1d_float,
            ctypes.c_int,
            self.array_2d_float,
            self.array_1d_float,
        ]

        self.math.compute_batched_field.restype = None
        self.math.compute_batched_field.argtypes = [
            ctypes.c_int,  # total_points
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # n_charges
            self.array_2d_float,  # x_0
            self.array_2d_float,  # x
            self.array_1d_float,  # Q
            self.array_2d_float,  # E
        ]

        self.math.compute_looped_field.restype = None
        self.math.compute_looped_field.argtypes = [
            ctypes.c_int,  # total_points
            ctypes.c_int,  # n_charges
            self.array_2d_float,  # x_0
            self.array_2d_float,  # x
            self.array_1d_float,  # Q
            self.array_2d_float,  # E
        ]

    def sparse_dot(self, A, B):
        # b is just a single vector, not a sparse matrix
        # a is a full sparse matrix
        res = np.zeros(len(A.data), dtype="float32")
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
        res = np.zeros(len(B.data), dtype="float32")
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
        res = np.zeros(len(A), dtype="float32")
        self.math.vecaddn.restype = None
        # self.math.vecaddn.restype = npct.ndpointer(
        #    dtype=self.array_1d_float, shape=len(A)
        # )
        self.math.vecaddn(res, A, B, len(A))
        return res

    def einsum_ij_i(self, A):
        res = np.zeros((A.shape[0]), dtype="float32")
        self.math.einsum_ij_i.restype = None
        self.math.einsum_ij_i(A.shape[0], A.shape[1], A, res)
        return res

    def einsum_ij_i_batch(self, A):
        # simply two vectors
        res = np.zeros((len(A), A[0].shape[0]), dtype="float32")
        self.math.einsum_ij_i_batch.restype = None
        self.math.einsum_ij_i_batch(len(A), A[0].shape[0], A[0].shape[1], A, res)
        res = res.reshape(res.shape[1], res.shape[0])
        return res

    def einsum_operation(self, R, r_mag, Q):
        res = np.zeros(3, dtype="float32")
        r_mag = r_mag.reshape(-1)
        R = R.reshape(r_mag.shape[0], 3)
        Q = Q.reshape(-1)
        self.math.einsum_operation.restype = None
        self.math.einsum_operation(
            len(Q),
            np.array(r_mag, dtype="float32"),
            np.array(Q, dtype="float32"),
            np.array(R, dtype="float32"),
            res,
        )
        return res

    def einsum_operation_batch(self, R, r_mag, Q, batch_size):
        # res = np.ascontiguousarray(np.zeros(3, dtype="float32"))
        # print("einsum in")
        res = np.zeros((batch_size, 3), dtype="float32")
        self.math.einsum_operation_batch.restype = None
        Q = Q.reshape(-1)
        self.math.einsum_operation_batch(
            batch_size,
            len(Q),
            np.array(r_mag, dtype="float32"),
            np.array(Q, dtype="float32"),
            np.array(R, dtype="float32"),
            res,
        )
        return res

    def compute_looped_field(self, x_0, x, Q):
        res = np.zeros_like(x_0, dtype="float32")
        self.math.compute_looped_field.restype = None
        Q = Q.reshape(-1)
        self.math.compute_looped_field(
            int(x_0.shape[0]),
            len(Q),
            np.array(x_0, dtype="float32"),
            np.array(x, dtype="float32"),
            np.array(Q, dtype="float32"),
            res,
        )

        return res

    def compute_batch_field(self, x_0, x, Q, batch_size):
        res = np.zeros_like(x_0, dtype="float32")
        self.math.compute_batched_field.restype = None
        Q = Q.reshape(-1)
        self.math.compute_batched_field(
            int(x_0.shape[0]),
            batch_size,
            len(Q),
            np.array(x_0, dtype="float32"),
            np.array(x, dtype="float32"),
            np.array(Q, dtype="float32"),
            res,
        )

        return res

    def thread_operation(self, x_0, n_iter, x, Q, step_size, dimensions):
        """
        Takes:
            x_0(array) - (3, 1) array of box position
            n_iter(int) - number of iterations of propagation for this slip
            x(np array) - positions of charges
            Q(np array) - charge values
            step_size(float) - step size of each step
            dimensions(array) - box limits
        Returns:
            res(array) - (2, ) array of curvature and distance
        """
        res = np.zeros(2, dtype="float32")
        n_charges = len(Q)
        self.math.thread_operation.restype = None
        Q = Q.reshape(-1)
        self.math.thread_operation(
            n_charges, n_iter, step_size, x_0, dimensions, x, Q, res
        )
        # print(res)

        return res

    def calc_esp_base(self, x_0, x, Q):
        """
        Takes:
            x_0(array) - (3, 1) array of box position
            x(np array) - positions of charges
            Q(np array) - charge values
        Returns:
            res(array) - (1, 1) array of electrostatic potential
        """
        res = np.zeros(1, dtype="float32")
        # self.math.efield.restype = None
        # Q = Q.reshape(-1)

        self.math.calc_esp_base(res, x_0, len(Q), x, Q.reshape(len(Q)))

        return res

    def calc_field_base(self, x_0, x, Q):
        """
        Takes:
            x_0(array) - (3, 1) array of box position
            x(np array) - positions of charges
            Q(np array) - charge values
        Returns:
            res(array) - (3, 1) array of electric field
        """
        res = np.zeros(3, dtype="float32")
        # self.math.efield.restype = None
        # Q = Q.reshape(-1)

        self.math.calc_field_base(res, x_0, len(Q), x, Q.reshape(len(Q)))

        return res

    def calc_field(self, x_0, x, Q):
        """
        Takes:
            x_0(array) - (3, 1) array of box position
            x(np array) - positions of charges
            Q(np array) - charge values
        Returns:
            res(array) - (3, 1) array of electric field
        """
        res = np.zeros(3, dtype="float32")
        # self.math.efield.restype = None
        # Q = Q.reshape(-1)

        self.math.calc_field(res, x_0, len(Q), x, Q.reshape(len(Q)))

        return res
