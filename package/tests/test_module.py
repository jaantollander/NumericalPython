"""Test for numerical routines using pytest and hypothesis.

Install nufft from http://github.com/dfm/python-nufft/
"""
from hypothesis import given
import hypothesis.strategies as st

import numpy as np
from nufft import nufft1 as nufft_fortran
from package.module import nudft


def test_nudft():
    x = 100 * np.random.random(1000)
    y = np.sin(x)

    Y1 = nudft(x, y, 1000)
    Y2 = nufft_fortran(x, y, 1000)

    assert np.allclose(Y1, Y2)
