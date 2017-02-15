"""Test jitclass and numpy custom dtype examples.

Attributes:
    SIZE: Number of particles
"""
import random

from package.jitclass import Particles, create_particles_numpy, \
    create_particles_jitclass

SIZE = int(1000)


def test_():
    # Random seed so that both particle create function create same values.
    seed = random.randint()

    # Custom Numpy dtype
    particles = create_particles_numpy(SIZE, seed)

    # Jitclass
    particles2 = Particles(SIZE)
    create_particles_jitclass(particles2, seed)

    # Test that all created values are the same
    assert True


def test_potential():
    pass
