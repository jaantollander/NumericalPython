"""Generated Jit"""
import numpy as np
import numba
from numba import float64

point2D = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
])

point3D = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
])


@numba.jit([float64(numba.from_dtype(point2D)),
            float64[:](numba.from_dtype(point2D)[:])],
           nopython=True, nogil=True, cache=True)
def distance2D(point):
    return np.sqrt(point.x ** 2 + point.y ** 2)


@numba.jit([float64(numba.from_dtype(point3D)),
            float64[:](numba.from_dtype(point3D)[:])],
           nopython=True, nogil=True, cache=True)
def distance3D(point):
    return np.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)


# FIXME
@numba.generated_jit(nopython=True, nogil=True)
def distance(point):
    if point.dtype is numba.from_dtype(point2D):
        return lambda point: np.sqrt(point.x ** 2 + point.y ** 2)
    elif point.dtype is numba.from_dtype(point3D):
        return lambda point: np.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
    else:
        raise Exception("Invalid argument")
