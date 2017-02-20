"""Tests for testing ahead-of-time complilation.

Todo:
    - More test functions
"""
import numba
from numba import void

from routines.examples.aot_compilation import create_extension
from routines.examples.structures import potential, particle_type, \
    create_random_particles_numpy


def test_aot_compilation():
    create_extension(extension_name='numba_ext',
                     exported_name='func',
                     signature=void(numba.typeof(particle_type)[:]),
                     function=potential)

    # We can now import the extension and run it
    from routines.examples.numba_ext import func
    particles = create_random_particles_numpy(10)
    assert func(particles) is None
