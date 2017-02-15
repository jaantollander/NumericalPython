"""Numerical routines written using ``Numpy`` and ``Numba``.

These examples are refined from or greatly influenced by the examples found from
github [#]_, [#]_, [#]_, [#]_.

References:
    .. [#] https://github.com/numba/numba/tree/master/examples
    .. [#] https://github.com/barbagroup/numba_tutorial_scipy2016
    .. [#] https://github.com/ContinuumIO/numbapro-examples
    .. [#] http://gmarkall.github.io/tutorials/pycon-uk-2015/#1

Notes:
    - Globals are treated as compile-time constants by Numba (using globals
      is not recommended).
    - Set envvar ``NUMBA_DISABLE_JIT=1`` to disable Numba compilation
      (for debugging).

"""
