"""Examples on how to use vectorize and guvectorize decorators and their
corresponding arguments.

Todo:
    - ``target=cuda`` -> cuda section
"""
import numpy as np
import numba
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
