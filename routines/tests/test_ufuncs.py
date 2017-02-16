import numpy as np
import pytest
from scipy.special import logit

from routines.examples.ufuncs import logit_serial, logit_par, matmul
from routines.utils import timefunc


@pytest.fixture(scope='module')
def args(size=int(1e5)):
    return np.random.random_sample(size),


@pytest.fixture(scope='module')
def correct(args):
    return logit(*args)


def test_logit(benchmark, correct, args):
    res = benchmark(logit, *args)
    assert np.allclose(res, correct)


def test_logit_serial(benchmark, correct, args):
    logit_serial(*args)
    res = benchmark(logit_serial, *args)
    assert np.allclose(res, correct)


def test_logit_par(benchmark, correct, args):
    logit_par(*args)
    res = benchmark(logit_par, *args)
    assert np.allclose(res, correct)


def test_gufunc(size=500):
    a = np.random.random((size, size))
    res = np.zeros_like(a)

    print()
    correct = timefunc(np.matmul, 'numpy.matmul', a, a)
    timefunc(matmul, 'numba matmul', a, a, res)

    assert np.allclose(res, correct)
