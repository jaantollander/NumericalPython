"""
Non-Uniform Fourier Transform

.. math::
   Y_k^\pm = \sum_{j=1}^N y(x_j) e^{\pm i k x_j}

Fast-Fourier Transform (FFT)

.. math::
   Y_k^\pm = \sum_{n=0}^{N-1} y_n e^{\pm i k n / N}

which requires that

- :math:`y_n` are the samples of function :math:`y_n = y(x_n)`
- :math:`x_n = x_0 + n \Delta x` are regular grid points

References:
    .. [#] https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/
    .. [#] http://adereth.github.io/blog/2013/11/29/colorful-equations/
    .. [#] http://www.cims.nyu.edu/cmcl/nufft/nufft.html

"""
import numpy as np
import numba


@numba.jit(nopython=True, nogil=True)
def nufftfreqs(M, df=1):
    """Compute the frequency range used in nufft for M frequency bins

    Args:
        M (int):
        df (float):

    Returns:
        numpy.ndarray:
    """
    return df * np.arange(-(M // 2), M - (M // 2))


# @numba.jit(nopython=True, nogil=True)
def nudft(x, y, M, df=1.0, iflag=1):
    """Non-Uniform Direct Fourier Transform

    Args:
        x:
        y:
        M:
        df:
        iflag:

    Returns:
        numpy.ndarray:
    """
    sign = -1 if iflag < 0 else 1
    return (1 / len(x)) * np.dot(y, np.exp(sign * 1j * nufftfreqs(M, df) * x[:, np.newaxis]))
