"""Test for numerical routines using pytest and hypothesis.

Attributes:
    NTHREADS (int):
        Number of threads to use.

    SIZE (int):
        Size of the test arrays.
"""
import numpy as np

from routines.examples.multithreading import make_multithread, inner_func_nb, func_np
from routines.examples.multithreading import make_singlethread
from routines.utils import timefunc

NTHREADS = 4
SIZE = int(1e6)


def test_nogil():
    func_nb = make_singlethread(inner_func_nb)
    func_nb_mt = make_multithread(inner_func_nb, NTHREADS)

    a = np.random.rand(SIZE)
    b = np.random.rand(SIZE)

    print()
    correct = timefunc(func_np, "numpy (1 thread)", a, b)
    res1 = timefunc(func_nb, "numba (1 thread)", a, b)
    res2 = timefunc(func_nb_mt, "numba (%d threads)" % NTHREADS, a, b)

    assert np.allclose(res1, correct), (res1, correct)
    assert np.allclose(res2, correct), (res2, correct)
