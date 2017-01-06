#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems: 
    nohup nice python my_experiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks

from real_ga import RealGeneticAlgorithms


argv = sys.argv[1:] # shortcut for input arguments

datapath = 'standard_ga' if len(argv) < 1 else argv[0]

dimensions = (2, 3, 5, 10, 20, 40) if len(argv) < 2 else eval(argv[1])
function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])  
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='Standard GA',
            comments='Standard genetic algorithm with 1-point mutation and crossover, mutation probability 0.05, crossover probability 0.95, \
            "rank" selection type and elitism=True')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes 
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10      # SET to zero if algorithm is entirely deterministic 


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation. 
    This implementation is an empty template to be filled 
    
    """
    # create GA instance
    gen_alg = RealGeneticAlgorithms(fun, optim='min')

    maxfunevals = min(1e8 * dim, maxfunevals)

     # initialize random population of GA
    popsize = min(maxfunevals, 200)
    alg.init_random_population(popsize, dim, (-5, 5))

    # run GA
    STANDARD_GA(gen_alg, maxfunevals, popsize, ftarget)

def STANDARD_GA(alg, maxfunevals, popsize, ftarget):
    fbest = np.inf

    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        if fbest > alg.population[-1].fitness_val:
            fbest = alg.population[-1].fitness_val
            xbest = alg.population[-1].individ
        if fbest < ftarget:  # task achieved 
            break

        # compute the next population
        alg.run(1)

    return xbest

t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim
