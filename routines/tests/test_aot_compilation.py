"""Tests for testing ahead-of-time complilation.

Todo:
    - More test functions
"""
import numba
from numba import void

from routines.examples.aot_compilation import create_extension
from routines.examples.structures import particle_type, \
    create_random_particles_numpy, distance


def test_aot_compilation():
    def potential(particles):
        """Calculate the potential at each particle using direct summation method.

        Arguments:
            particles:
                The list of particles

        """
        for i in range(particles.size):
            for j in range(i, particles.size):
                dx = particles.position.x[i] - particles.position.x[j]
                dy = particles.position.y[i] - particles.position.y[j]
                dz = particles.position.z[i] - particles.position.z[j]
                r = distance(dx, dy, dz)
                if r != 0.0:
                    particles.phi[i] += particles.m[i] / r
                    particles.phi[j] += particles.m[j] / r

    create_extension(extension_name='numba_ext',
                     exported_name='func',
                     signature=void(numba.typeof(particle_type)[:]),
                     function=potential)

    # We can now import the extension and run it
    from routines.examples.numba_ext import func
    particles = create_random_particles_numpy(10)
    assert func(particles) is None
