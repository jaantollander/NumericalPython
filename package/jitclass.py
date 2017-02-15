"""Jitclass and Numpy custom dtype examples

This example implements 3D point as both, numpy custom dtype, and jitclass.
"""
import numpy as np
import numba


point_type = np.dtype({
    'names': ['x', 'y', 'z'],
    'formats': [np.double, np.double, np.double]
})


@numba.jitclass([
    ('x', numba.float64),
    ('y', numba.float64),
    ('z', numba.float64),
])
class Point(object):
    r"""Structure representing individual point in 3D space

    Arguments:
        x, y, z (float): coordinates of the point

    Attributes:
        x, y, z (float): coordinates of the point
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other):
        return ((self.x - other.x) ** 2 +
                (self.y - other.y) ** 2 +
                (self.z - other.z) ** 2) ** .5
