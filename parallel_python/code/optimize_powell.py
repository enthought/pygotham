#!/usr/bin/env python
"""
Minimize the selected model with Powell's method.

Requires: development version of mystic, pathos, pyina
  http://pypi.python.org/pypi/mystic
  http://pypi.python.org/pypi/pathos
  http://pypi.python.org/pypi/pyina
"""

def optimize(solver, mapper, nodes, target='rosen', **kwds):
    if target == 'rosen': # 3d-rosenbrock
        # Rosenbrock function
        from dejong import rosen as the_model
        ndim = 3
        actual_coeffs = [1.0] * ndim
        pprint = list
    else: # 4th-order chebyshev
        # Chebyshev cost function
        from poly import chebyshev4cost as the_model
        from poly import chebyshev4coeffs as actual_coeffs
        ndim = len(actual_coeffs)
        from mystic.math import poly1d as pprint

    # number of trials
    N = nodes
    print "Number of trials: %s" % N
    print "==============="

    # initial guess
    import random
    x0 = ([random.uniform(-100,100) for i in xrange(ndim)] for i in xrange(N))
    model = (the_model for i in xrange(N))

    # minimize the function
    results = mapper(the_solver, model, x0, **kwds)

    # find the results with the lowest energy
    from optimize_helper import best_results
    solution = best_results(results)

    print "==============="
    print "Actual params:\n %s" % pprint(actual_coeffs)
    print "Solved params:\n %s" % pprint(solution[0])
    print "Function value: %s" % solution[1]
    print "Total function evals: %s" % solution[4]
    return 

# Powell's Directonal solver
from optimize_helper import fmin_powell as the_solver

# get the map functions
from itertools import imap                   # itertools
from pathos.pp_map import pp_map             # parallelpython
from pathos.mp_map import mp_map             # multiprocessing
from pyina.ez_map import ez_map2 as mpi_map  # mpi4py


if __name__ == '__main__':
    target = 'rosen'
   #target = 'cheby'
    print "Function: %s" % target
    print "Solver: %s" % 'fmin_powell'
   #NOTE: some of the below should fail, due to how objects are shipped in map
    optimize(the_solver, imap, nodes=1, target=target)
   #optimize(the_solver, pp_map, nodes=1, target=target, ncpus=1)
   #optimize(the_solver, mpi_map, nodes=1, target=target, nnodes=1)
   #optimize(the_solver, mp_map, nodes=1, target=target, nproc=1)

 
# end of file
