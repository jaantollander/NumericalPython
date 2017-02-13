import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def routine():
    """Numerical routine

    LaTeX Eauation

    .. math::
       \exp(i \pi} = -1

    """
    return
