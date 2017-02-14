"""Test for numerical routines using pytest and hypothesis.

Attributes:
    NTHREADS:
        Number of threads to use.

    SIZE:
        Size of the test arrays.
"""
from timeit import repeat
import numpy as np
from hypothesis import given

from package.nogil import make_multithread, inner_func_nb, func_np
from package.nogil import make_singlethread

NTHREADS = 4
SIZE = 1e6


def timefunc(correct, s, func, *args, **kwargs):
    """Benchmark *func* and print out its runtime.

    Args:
        correct:
        s:
        func:
        *args:
        **kwargs:

    Returns:
        object:
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                          number=5, repeat=2)) * 1000))
    return res


def test_nogil():
    func_nb = make_singlethread(inner_func_nb)
    func_nb_mt = make_multithread(inner_func_nb, NTHREADS)

    a = np.random.rand(SIZE)
    b = np.random.rand(SIZE)

    correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
    timefunc(correct, "numba (1 thread)", func_nb, a, b)
    timefunc(correct, "numba (%d threads)" % NTHREADS, func_nb_mt, a, b)
