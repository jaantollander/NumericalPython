"""Benchmarks of the functions"""
import matplotlib.pyplot as plt
import numpy as np

from routines.examples.multithreading import (
    make_multithread, inner_func_nb, make_singlethread
)
from routines.examples.ufuncs import logit_serial, logit_par
from routines.utils import benchmark


def ufuncs():
    """Benchmark ufuncs"""

    def args(size):
        return np.random.random_sample(size),

    fig, ax = plt.subplots()
    ax.plot(*benchmark(logit_serial, args), label='Serial Float64')
    ax.plot(*benchmark(logit_par, args), label='Parallel Float64')
    plt.legend()
    plt.show()


def multithreading(nthreads=4):
    """Benchmark multithreading

    Args:
        nthreads (int):
    """
    funcs = [(1, make_singlethread(inner_func_nb))] + \
            [(n, make_multithread(inner_func_nb, 2)) for n in range(nthreads)]

    def args(size):
        a = np.random.rand(size)
        b = np.random.rand(size)
        return a, b

    fig, ax = plt.subplots()

    for i, func in funcs:
        ax.plot(*benchmark(func, args), label="%d thread(s)" % nthreads)

    plt.legend()
    plt.show()
