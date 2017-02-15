"""Test jitclass and numpy custom dtype examples.

Attributes:
    SIZE: Number of particles
"""
import numpy as np

from package.jitclass import Particles, create_particles

SIZE = int(1000)


def test_custom_dtype():
    particles = create_particles(SIZE)
    assert True


def test_jitclass():
    particles = Particles(SIZE)
    assert True


def test_potential():
    pass
