"""Test generated_jit"""
import numpy as np

from routines.examples.generated_jit import distance2D, distance3D, point2D, point3D, distance


def test_distance2D():
    point = np.zeros(10, point2D)
    dists = distance2D(point)
    assert isinstance(dists, np.ndarray)

    for i, p in enumerate(point):
        dist = distance2D(p)
        assert dists[i] == dist
        assert isinstance(dist, float)


def test_distance3D():
    point = np.zeros(10, point3D)
    dists = distance3D(point)
    assert isinstance(dists, np.ndarray)

    for i, p in enumerate(point):
        dist = distance3D(p)
        assert dists[i] == dist
        assert isinstance(dist, float)


def test_generated_distance():
    point = np.zeros(10, point2D)
    distance(point)
    assert True
