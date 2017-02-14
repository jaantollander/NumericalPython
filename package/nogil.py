"""Example of multithreaded routine using ``nogil=True``.

Test function in these examples is

.. math::
   f(a, b) = \exp(2.1 a + 3.2 b)

"""
from __future__ import print_function, division, absolute_import

import math
import threading

import numpy as np
from numba import jit


def func_np(a, b):
    r"""Control function using Numpy.

    Args:
        a (float|np.ndarray): :math:`a`
        b (float|np.ndarray): :math:`b`

    Returns:
        float|np.ndarray: :math:`f(a, b)`
    """
    return np.exp(2.1 * a + 3.2 * b)


@jit('void(double[:], double[:], double[:])', nopython=True, nogil=True)
def inner_func_nb(result, a, b):
    """Function under test.

    Args:
        result (np.ndarray):
            Result array where the results from function :math:`f(a, b)` are
            stored.
        a (np.ndarray): :math:`a`
        b (np.ndarray): :math:`b`
    """
    for i in range(len(result)):
        result[i] = math.exp(2.1 * a[i] + 3.2 * b[i])


def make_singlethread(inner_func):
    """Run the given function inside a single thread.

    Args:
        inner_func:

    Returns:
        object:
    """

    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        inner_func(result, *args)
        return result

    return func


def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.

    Args:
        inner_func:
        numthreads (int):

    Returns:
        object:
    """

    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
                  for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result

    return func_mt
