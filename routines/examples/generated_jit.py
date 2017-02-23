"""Generated Jit"""
import numpy as np
import numba

point2D = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
])

point3D = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
])


def create_random_point(d):
    if d == 2:
        return np.zeros(1, dtype=point2D)
    elif d == 3:
        return np.zeros(1, dtype=point3D)


def distance2D(point):
    return np.sqrt(point.x ** 2 + point.y ** 2)


def distance3D(point):
    return np.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)


@numba.generated_jit(nopython=True, nogil=True)
def distance(point):
    if isinstance(point, numba.from_dtype(point2D)[:]):
        return distance2D
    elif isinstance(point, numba.from_dtype(point3D)[:]):
        return distance2D
    else:
        raise Exception("Invalid argument")
