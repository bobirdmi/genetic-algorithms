#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs the timing experiment for PURE_RANDOM_SEARCH.

CAPITALIZATION indicates code adaptations to be made
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

"""

import time
import numpy as np
import fgeneric
import bbobbenchmarks as bn

from geneticalgs import RealGA, DiffusionGA, MigrationGA


# MAX_FUN_EVALS = '1e3 + 1e6/dim' # per run, adjust to default but prevent very long runs (>>30s)
MAX_FUN_EVALS = '1e3 + 1e5/dim'

def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation. 
    This implementation is an empty template to be filled 
    
    """
    # uncomment for run a standard GA
    #gen_alg = RealGA(fun, optim='min')

    # uncomment the code below and the previous one for run a diffusion GA
    #gen_alg = DiffusionGA(gen_alg)

    # uncomment for run a migration GA
    gen_alg1 = RealGA(fun, optim='min')
    gen_alg2 = RealGA(fun, optim='min')    

    maxfunevals = min(1e8 * dim, maxfunevals)

     # initialize random population of GA
    popsize = min(maxfunevals, 200)

    # uncomment for standard or diffusion GA
    #gen_alg.init_random_population(popsize, dim, (-5, 5))

    # uncomment for run a migration GA
    one_part = popsize // 2
    gen_alg1.init_random_population(one_part, dim, (-5, 5))
    gen_alg2.init_random_population(popsize - one_part, dim, (-5, 5))
    mga = MigrationGA()
    mga.init_populations([gen_alg1, gen_alg2])

    # run GA
    # uncomment for run standard and diffusion GA
    #RUN_GA(gen_alg, maxfunevals, popsize, ftarget)

    # run Migration GA
    # uncomment for run a migration GA
    RUN_MGA(mga, maxfunevals, popsize, ftarget)

def RUN_MGA(alg, maxfunevals, popsize, ftarget):
    fbest = np.inf

    for i in range(0, int(np.ceil(maxfunevals / popsize))):
	# compute the next population
	if i % 4 == 0 and i > 0:  # performs migration every 5 generations
	    migrate = True
        else:
	    migrate = False

        _, solution = alg.run(1, period=1, migrant_num=3, cloning=True, migrate=migrate)

        if fbest > solution[1]:
            fbest = solution[1]
            xbest = solution[0]

        if fbest < ftarget:  # task achieved 
            break

    return xbest

def RUN_GA(alg, maxfunevals, popsize, ftarget):
    fbest = np.inf

    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        if fbest > alg.best_solution[1]:
            fbest = alg.best_solution[1]
            xbest = alg.best_solution[0]

        if fbest < ftarget:  # task achieved 
            break

        # compute the next population
        alg.run(1)

    return xbest

timings = []
runs = []
dims = []
for dim in (2, 3, 5, 10, 20, 40, 80, 160):  # 320, 640, 1280, 2560, 5120, 10240, 20480):
    nbrun = 0
    f = fgeneric.LoggingFunction('tmp').setfun(*bn.instantiate(8, 1))
    t0 = time.time()
    while time.time() - t0 < 30: # at least 30 seconds
        run_optimizer(f.evalfun, dim, eval(MAX_FUN_EVALS), f.ftarget)  # adjust maxfunevals
        nbrun = nbrun + 1
    timings.append((time.time() - t0) / f.evaluations)
    dims.append(dim)    # not really needed
    runs.append(nbrun)  # not really needed
    f.finalizerun()
    print '\nDimensions:',
    for i in dims:
        print ' %11d ' % i,
    print '\n      runs:',
    for i in runs:
        print ' %11d ' % i,
    print '\n times [s]:',
    for i in timings:
        print ' %11.1e ' % i, 
    print ''

