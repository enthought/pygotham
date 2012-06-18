#!/usr/bin/env python
"""example: using the same code with different parallel backends

Requires: development version of pathos, pyina
  http://pypi.python.org/pypi/pathos
  http://pypi.python.org/pypi/pyina

Run with:
>$ python all_scatter_gather.py
"""

import numpy as np
from pyina.ez_map import ez_map2 as mpi_map  # mpi4py
from pathos.mp_map import mp_map             # multiprocessing
from pathos.pp_map import pp_map             # parallelpython
nodes = 2; N = 3

# take sin squared of all data
def sin2(xi):
    """sin squared of all data"""
    import numpy as np
    return np.sin(xi)**2


# print the input to screen
x = np.arange(N * nodes, dtype=np.float64)
print("Input: %s\n" % x)


# run sin2 in series, then print to screen
print("Running serial python ...")
y = map(sin2, x)
print("Output: %s\n" % np.asarray(y))


# map sin2 to the workers, then print to screen
print("Running mpi4py on %d cores..." % nodes)
y = mpi_map(sin2, x, nnodes=nodes)
print("Output: %s\n" % np.asarray(y))


# map sin2 to the workers, then print to screen
print("Running multiprocesing on %d processors..." % nodes)
y = mp_map(sin2, x, nproc=nodes)
print("Output: %s\n" % np.asarray(y))


# map sin2 to the workers, then print to screen
print("Running parallelpython on %d cpus..." % nodes)
y = pp_map(sin2, x, ncpus=nodes, servers=('mycpu.mydomain.com',))
print("Output: %s\n" % np.asarray(y))

# EOF
