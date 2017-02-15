import matplotlib.pyplot as plt
import numpy as np

from package.ufuncs import logit_serial, logit_par
from package.utils import benchmark


def arg_gen64(size):
    return np.random.random_sample(size),


fig, ax = plt.subplots()
ax.plot(*benchmark(logit_serial, arg_gen64),
        label='Serial Float64')
ax.plot(*benchmark(logit_par, arg_gen64),
        label='Parallel Float64')
plt.legend()
plt.show()
