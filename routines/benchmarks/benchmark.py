"""Benchmarks of the functions"""
import matplotlib.pyplot as plt
import numpy as np

from routines.multithreading import make_multithread, inner_func_nb
from routines.multithreading import make_singlethread
from routines.ufuncs import logit_serial, logit_par
from routines.utils import benchmark

NTHREADS = 4


def benchmark_logit():
    def arg_gen(size):
        return np.random.random_sample(size),

    fig, ax = plt.subplots()
    ax.plot(*benchmark(logit_serial, arg_gen),
            label='Serial Float64')
    ax.plot(*benchmark(logit_par, arg_gen),
            label='Parallel Float64')
    plt.legend()
    plt.show()


def benchmark_nogil():
    func_nb = make_singlethread(inner_func_nb)
    func_nb_mt = make_multithread(inner_func_nb, NTHREADS)

    def arg_gen(size):
        a = np.random.rand(size)
        b = np.random.rand(size)
        return a, b

    fig, ax = plt.subplots()
    ax.plot(*benchmark(func_nb, arg_gen),
            label='1 thread')
    ax.plot(*benchmark(func_nb_mt, arg_gen),
            label="%d threads" % NTHREADS)
    plt.legend()
    plt.show()
