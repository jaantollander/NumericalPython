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


# -------------------------------------------------------------------------
# Optimized version
# -------------------------------------------------------------------------


@numba.jit(nopython=True)
def build_grid_fast(x, c, tau, Msp, ftau, E3):
    """Build grid fast

    Args:
        x:
        c:
        tau:
        Msp:
        ftau:
        E3:

    Returns:
        numpy.ndarray:
    """
    Mr = ftau.shape[0]
    hx = 2 * np.pi / Mr

    # precompute some exponents
    for j in range(Msp + 1):
        E3[j] = np.exp(-(np.pi * j / Mr) ** 2 / tau)

    # spread values onto ftau
    for i in range(x.shape[0]):
        xi = x[i] % (2 * np.pi)
        m = 1 + int(xi // hx)
        xi = (xi - hx * m)
        E1 = np.exp(-0.25 * xi ** 2 / tau)
        E2 = np.exp((xi * np.pi) / (Mr * tau))
        E2mm = 1
        for mm in range(Msp):
            ftau[(m + mm) % Mr] += c[i] * E1 * E2mm * E3[mm]
            E2mm *= E2
            ftau[(m - mm - 1) % Mr] += c[i] * E1 / E2mm * E3[mm + 1]
    return ftau


def nufft_numba_fast(x, c, M, df=1.0, eps=1E-15, iflag=1):
    """Fast Non-Uniform Fourier Transform with Numba

    Args:
        x:
        c:
        M:
        df:
        eps:
        iflag:

    Returns:
        numpy.ndarray:
    """
    Msp, Mr, tau = _compute_grid_params(M, eps)
    N = len(x)

    # Construct the convolved grid
    ftau = build_grid_fast(x * df, c, tau, Msp,
                           np.zeros(Mr, dtype=c.dtype),
                           np.zeros(Msp + 1, dtype=x.dtype))

    # Compute the FFT on the convolved grid
    if iflag < 0:
        Ftau = (1 / Mr) * np.fft.fft(ftau)
    else:
        Ftau = np.fft.ifft(ftau)
    Ftau = np.concatenate([Ftau[-(M // 2):], Ftau[:M // 2 + M % 2]])

    # Deconvolve the grid using convolution theorem
    k = nufftfreqs(M)
    return (1 / N) * np.sqrt(np.pi / tau) * np.exp(tau * k ** 2) * Ftau