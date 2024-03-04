import numpy as np
from CPET.utils.fastmath import nb_subtract, power, nb_norm, nb_cross
from CPET.utils.c_ops import Math_ops
#math = Math_ops(shared_loc="../utils/math_module.so")

class calculator:
    def __init__(self, math_loc="../utils/math_module.so"):
        self.math = Math_ops(shared_loc=math_loc)

    def __call__(self, x_0, x, Q):

        # Create matrix R
        R = nb_subtract(x_0, x)
        R_sq = R**2
        r_mag_sq = self.math.einsum_ij_i(R_sq).reshape(-1, 1)
        r_mag_cube = np.power(r_mag_sq, 3 / 2)
        E = self.math.einsum_operation(R, 1 / r_mag_cube, Q)

        return E
