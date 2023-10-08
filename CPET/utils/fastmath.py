import numpy as np
from numba import jit

"""
Making basic math faster!!
"""


def power(a,b):
    """
    Computes binary exponentiation
    Takes
        a(float or array) - a number or array of shape (N,) or (N,1)
        b(float) - a real positive number
    Returns
        out_(float or array) - element-wise exponentiation of a by b, a number or array of shape (N,) or (N,1)
    """
    if b==0:
        return 1
    out_ = power(a,b//2)
    out_ = out_ * out_
    if b%2 !=0 :
        out_ = out_ * a
    return out_

@jit(nopython=True)
def nb_subtract(a,b):
    """
    Computes subtraction
    Takes
        a(array or array-like)
        b(array or array-like)
    Returns
        np.subtract(a,b) - difference between a and b - jitted!
    """
    return np.subtract(a,b)
