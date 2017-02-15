import numpy as np
from scipy.special import logit

from package.ufuncs import logit_serial, logit_par, matmul
from package.utils import timefunc


SIZE = int(1e5)


def test_ufunc():
    a = np.random.random_sample(SIZE)

    print()
    correct = timefunc(logit, 'Scipy', a)
    res1 = timefunc(logit_serial, 'Serial', a)
    res2 = timefunc(logit_par, 'Parallel', a)

    assert np.allclose(res1, correct)
    assert np.allclose(res2, correct)


def test_gufunc():
    a = np.random.random((500, 500))
    res = np.zeros_like(a)

    print()
    correct = timefunc(np.matmul, 'numpy.matmul', a, a)
    timefunc(matmul, 'numba matmul', a, a, res)

    assert np.allclose(res, correct)
