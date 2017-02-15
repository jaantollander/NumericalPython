"""Examples on how to use vectorize and guvectorize decorators and their
corresponding arguments."""
import numpy as np
import numba


@numba.vectorize()
def function():
    pass
