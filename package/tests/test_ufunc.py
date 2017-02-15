import numpy as np
from scipy.special import logit

from package.ufuncs import logit_serial, logit_par
from package.utils import timefunc


SIZE = int(1e5)


def test_logit():
    a = np.random.random_sample(SIZE)

    correct = timefunc(logit, 'Scipy', a)
    res1 = timefunc(logit_serial, 'Serial', a)
    res2 = timefunc(logit_par, 'Parallel', a)

    assert np.allclose(res1, correct)
    assert np.allclose(res2, correct)
