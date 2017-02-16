"""Structured data

- Numpy custom dtypes
- Jitclass

This example implements particle in 3D space as both, numpy custom dtype, and
jitclass.

Attributes:
    particle_type (numpy.dtype):

Todo:
    - Fix Jitclass which is slower due to array access order
"""
import numpy as np
import numba
from numba.types import optional, int64, float64

# --------------------------------------
# Numpy custom dtype version of 3D point
# --------------------------------------

particle_type = np.dtype({
    'names': ['x', 'y', 'z', 'm', 'phi'],
    'formats': [np.float64, np.float64, np.float64, np.float64, np.float64]
})


@numba.jit(nopython=True, locals={'seed': optional(int64)})
def create_random_particles_numpy(size, seed=None):
    """Create size number of random particles

    Args:
        size (int):
            Number of particles

        seed (int):
            Seed for numpy random generator

    """
    if seed is not None:
        np.random.seed(seed)

    particles = np.empty(size, dtype=particle_type)
    for particle in particles:
        particle.x = np.random.uniform(-1.0, 1.0)
        particle.y = np.random.uniform(-1.0, 1.0)
        particle.z = np.random.uniform(-1.0, 1.0)
        particle.m = np.random.uniform(0.0, 1.0)
        particle.phi = 0.0
    return particles


# ----------------------------------
# Numba jitclass version of 3D point
# ----------------------------------

@numba.jitclass([
    ('size', numba.int64),
    ('x', numba.float64[:]),
    ('y', numba.float64[:]),
    ('z', numba.float64[:]),
    ('m', numba.float64[:]),
    ('phi', numba.float64[:]),
])
class Particles(object):
    r"""Structure representing multiple particles in 3D space.

    Arguments:
        size (int):
            Number of particles.

    Attributes:
        x, y, z (ndarray):
            Coordinates of the point

        m (ndarray):
            Mass of the particle

        phi (ndarray):
            The potential of the particle

    """

    def __init__(self, size):
        self.size = size
        self.x = np.zeros(self.size)
        self.y = np.zeros(self.size)
        self.z = np.zeros(self.size)
        self.m = np.zeros(self.size)
        self.phi = np.zeros(self.size)


@numba.jit(nopython=True, locals={'seed': optional(int64)})
def create_random_particles_jitclass(particles, seed=None):
    """Create random particles

    Args:
        seed (int):
    """
    if seed is not None:
        np.random.seed(seed)

    for i in range(particles.size):
        particles.x[i] = np.random.uniform(-1.0, 1.0)
        particles.y[i] = np.random.uniform(-1.0, 1.0)
        particles.z[i] = np.random.uniform(-1.0, 1.0)
        particles.m[i] = np.random.uniform(0.0, 1.0)
        particles.phi[i] = 0.0


# ---------------------
# Interaction potential
# ---------------------

@numba.jit(nopython=True, nogil=True)
def distance(dx, dy, dz):
    r"""Euclidean distance between two points.

    .. math::
        \sqrt{(\Delta x^{2} + \Delta y^{2} + \Delta z^{2})}

    Args:
        dx (float):
        dy (float):
        dz (float):

    Returns:
        float:
    """
    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5


@numba.jit(nopython=True, nogil=True)
def potential(particles):
    """Calculate the potential at each particle using direct summation method.

    Arguments:
        particles (Iterable[point_type]|Particles):
            The list of particles

    """
    for i in range(particles.size):
        for j in range(i, particles.size):
            dx = particles.x[i] - particles.x[j]
            dy = particles.y[i] - particles.y[j]
            dz = particles.z[i] - particles.z[j]
            r = distance(dx, dy, dz)
            if r != 0.0:
                particles.phi[i] += particles.m[i] / r
                particles.phi[j] += particles.m[j] / r


@numba.jit(nopython=True)
def reset_potential(particles):
    """Set all potential values to zero."""
    particles.phi[:] = 0.0
