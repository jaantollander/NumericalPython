import numpy as np
from scipy.special import logit

from routines.examples.ufuncs import logit_serial, logit_par, matmul
from routines.utils import timefunc


def test_ufunc(size=int(1e5)):
    a = np.random.random_sample(size)

    print()
    correct = timefunc(logit, 'Scipy', a)
    res1 = timefunc(logit_serial, 'Serial', a)
    res2 = timefunc(logit_par, 'Parallel', a)

    assert np.allclose(res1, correct)
    assert np.allclose(res2, correct)


def test_gufunc(size=500):
    a = np.random.random((size, size))
    res = np.zeros_like(a)

    print()
    correct = timefunc(np.matmul, 'numpy.matmul', a, a)
    timefunc(matmul, 'numba matmul', a, a, res)

    assert np.allclose(res, correct)
