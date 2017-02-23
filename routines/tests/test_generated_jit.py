"""Test generated_jit"""
from routines.examples.generated_jit import distance, create_random_point


def test_distance2D():
    point = create_random_point(2)
    dist = distance(point)
    assert isinstance(dist, float)


def test_distance3D():
    point = create_random_point(3)
    dist = distance(point)
    assert isinstance(dist, float)
