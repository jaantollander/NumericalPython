import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def routine(arg1, arg2):
    """Numerical routine

    Write your equations in LaTeX like this

    .. math::
       \exp(i \pi} = -1

    Args:
        arg1 (float):
        arg2 (float):

    Returns:
        float:

    """
    return
