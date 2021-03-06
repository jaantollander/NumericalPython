"""Test jitclass and numpy custom dtype examples."""
import random

import numpy as np
import pytest

from routines.examples.structures import Particles, create_random_particles_numpy, \
    create_random_particles_jitclass, reset_potential, potential2, \
    potential1

SEED = random.randint(0, 100)
SIZE = 1000


@pytest.fixture(scope='module')
def particles_custom_dtype(size=SIZE, seed=SEED):
    return create_random_particles_numpy(size, seed)


@pytest.fixture(scope='module')
def particles_jitclass(size=SIZE, seed=SEED):
    particles2 = Particles(size)
    create_random_particles_jitclass(particles2, seed)
    return particles2


def test_compare_data(particles_custom_dtype, particles_jitclass):
    """Validates that both structures create same output."""
    potential1(particles_custom_dtype)
    potential2(particles_jitclass)

    assert np.allclose(particles_custom_dtype['position']['x'], particles_jitclass.x)
    assert np.allclose(particles_custom_dtype['position']['y'], particles_jitclass.y)
    assert np.allclose(particles_custom_dtype['position']['z'], particles_jitclass.z)
    assert np.allclose(particles_custom_dtype['m'], particles_jitclass.m)
    assert np.allclose(particles_custom_dtype['phi'], particles_jitclass.phi)

    reset_potential(particles_custom_dtype)
    reset_potential(particles_jitclass)


def test_custom_dtype(benchmark, particles_custom_dtype):
    potential1(particles_custom_dtype)
    benchmark(potential1, particles_custom_dtype)


def test_jitclass(benchmark, particles_jitclass):
    potential2(particles_jitclass)
    benchmark(potential2, particles_jitclass)
