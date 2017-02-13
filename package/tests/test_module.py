"""Test for numerical routines using pytest and hypothesis.

Install nufft from http://github.com/dfm/python-nufft/
"""

from time import time

import numpy as np
from nufft import nufft1 as nufft_fortran

from package.module import nudft


def cmp_nufft(nufft_func, M=1000, Mtime=100000):
    # Test vs the direct method
    print(30 * '-')
    name = {'nufft1': 'nufft_fortran'}.get(nufft_func.__name__,
                                           nufft_func.__name__)
    print("testing {0}".format(name))

    rng = np.random.RandomState(0)
    x = 100 * rng.rand(M + 1)
    y = np.sin(x)
    for df in [1, 2.0]:
        for iflag in [1, -1]:
            F1 = nudft(x, y, M, df=df, iflag=iflag)
            F2 = nufft_func(x, y, M, df=df, iflag=iflag)
            assert np.allclose(F1, F2)

    print("- Results match the DFT")

    # Time the nufft function
    # x = 100 * rng.rand(Mtime)
    # y = np.sin(x)
    # times = []
    # for i in range(5):
    #     t0 = time()
    #     F = nufft_func(x, y, Mtime)
    #     t1 = time()
    #     times.append(t1 - t0)
    #
    # print(
    #     "- Execution time (M={0}): {1:.2g} sec".format(Mtime, np.median(times)))


def test_nudft():
    x = 100 * np.random.random(1000)
    y = np.sin(x)

    Y1 = nudft(x, y, 1000)
    Y2 = nufft_fortran(x, y, 1000)

    assert np.allclose(Y1, Y2)
