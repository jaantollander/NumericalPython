"""Test jitclass and numpy custom dtype examples.

Attributes:
    SIZE: Number of particles
"""
import random
import numpy as np

from package.jitclass import Particles, create_particles_numpy, \
    create_particles_jitclass, potential, reset_potential

SIZE = int(1000)


def test_structured_data():
    # Random seed so that both particle create function create same values.
    seed = random.randint(0, 100)

    # Custom Numpy dtype
    particles = create_particles_numpy(SIZE, seed)

    # Jitclass
    particles2 = Particles(SIZE)
    create_particles_jitclass(particles2, seed)

    # Test that all created values are the same
    assert np.allclose(particles['x'], particles2.x)
    assert np.allclose(particles['y'], particles2.y)
    assert np.allclose(particles['z'], particles2.z)
    assert np.allclose(particles['m'], particles2.m)

    # Compute potentials
    potential(particles)
    potential(particles2)

    # Assert that computed potentials are equal
    assert np.allclose(particles['phi'], particles2.phi)

    reset_potential(particles)
    reset_potential(particles2)
