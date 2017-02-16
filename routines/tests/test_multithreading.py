"""Test for numerical routines using pytest and hypothesis.

Args:
    size:
        Size of the test arrays.

    nthreads (int):
        Number of threads to use.
"""
import numpy as np
import pytest

from routines.examples.multithreading import make_multithread, inner_func_nb, func_np
from routines.examples.multithreading import make_singlethread


@pytest.fixture(scope='module')
def args(size=int(1e6)):
    """Arguments for the functions that are tested. Scope is set to module
    so that all functions are tested with same arguments."""
    a = np.random.rand(size)
    b = np.random.rand(size)
    return a, b


@pytest.fixture(scope='module')
def correct(args):
    return func_np(*args)


@pytest.fixture()
def func_nb():
    return make_singlethread(inner_func_nb)


@pytest.fixture()
def func_nb_mt(nthreads=4):
    return make_multithread(inner_func_nb, nthreads)


def test_func_np(benchmark, args):
    benchmark(func_np, *args)


def test_func_nb(benchmark, correct, func_nb, args):
    func_nb(*args)
    res = benchmark(func_nb, *args)
    assert np.allclose(res, correct)


def test_func_nb_mt(benchmark, correct, func_nb_mt, args):
    func_nb_mt(*args)
    res = benchmark(func_nb_mt, *args)
    assert np.allclose(res, correct)
