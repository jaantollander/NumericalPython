"""Function for testing, profiling and benchmarking numba routines.

Todo:
    - Bokeh plot benchmark output
"""
import math
import sys
from timeit import repeat
import numpy as np


def format_time(timespan, precision=3):
    """Formats the timespan in a human readable form

    Args:
        timespan (float):
            Time in seconds.
        precision (int):
            Desired precision.
    """

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover %= length
                time.append(u'%s%s' % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = [u"s", u"ms", u'us', "ns"]  # the recordable value
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms", u'\xb5s', "ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    # return u"%.*g %s" % (precision, timespan * scaling[order], units[order])
    return u"{:.1f} {}".format(timespan * scaling[order], units[order])


def timefunc(func, msg, *args, **kwargs):
    """Benchmark *func* and print out its runtime.

    Args:
        msg (str):
        func:
        *args:
        **kwargs:

    Returns:
        object:
    """

    # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)

    # Timeit
    print(msg.ljust(20), end=" ")
    timespan = min(repeat(lambda: func(*args, **kwargs), number=5, repeat=2))
    print(format_time(timespan))
    return res


def benchmark(func, arg_gen, num=500, seed=None):
    """Benchmark a function

    Args:
        func:
            Function to benchmark

        arg_gen:
        num (int):
        seed (int, optional):

    Returns:
        (ndarray, ndarray):

    Todo:
        - float32, float64
        - Value range
        - generating argument to functions
    """
    np.random.seed(seed)

    sizes = np.logspace(1.0, 6.0, num, dtype=np.int64)
    times = np.zeros_like(sizes, dtype=np.float64)

    # Make sure the function is compiled before we start the benchmark
    args = arg_gen(10)
    res = func(*args)

    for i, size in enumerate(sizes):
        print(i)
        args = arg_gen(size)
        times[i] = min(repeat(lambda: func(*args), number=5, repeat=2))

    return sizes, times
