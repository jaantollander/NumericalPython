"""Examples on how to use vectorize and guvectorize decorators and their
corresponding arguments.

Todo:
    - Improve guvectorize example
    - ``target=cuda`` -> cuda section
"""
import numba
import numpy as np
from numba import float32, float64


def logit(a):
    """Logit

    .. math::
       f(a) = \log \left(\frac{a}{1-a}\right)

    Returns:
        float:
    """
    return np.log(a / (1 - a))


signature = [
    float32(float32),
    float64(float64)
]
logit_serial = numba.vectorize(signature)(logit)
logit_par = numba.vectorize(signature, target='parallel')(logit)


# ------------------
# Generalized ufuncs
# ------------------

@numba.guvectorize('float64[:, :], float64[:, :], float64[:, :]',
                   '(m,n),(n,p)->(m,p)')
def matmul(A, B, C):
    """Matrix multiplication"""
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
