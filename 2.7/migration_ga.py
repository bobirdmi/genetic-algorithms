# Copyright 2017 Dmitriy Bobir <bobirdima@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import numpy
import copy
from itertools import izip


class MigrationGA(object):
    u"""
    This class implements migration model of GA, namely island model (not stepping-stone).
    It works with binary or real GA.

    Attributes:
        type (str): Type of used genetic algorithms: may be 'binary' or 'real'.

    You may initialize instance of this class the following way

    .. testcode::

       from geneticalgs import RealGA, MigrationGA
       import math

       # define some function whose global minimum or maximum we are searching for
       # this function takes as input one-dimensional number
       def fitness_function(x):
           # the same function is used in examples
           return abs(x*(math.sin(x/11)/5 + math.sin(x/110)))

       # initialize two or more standard real GAs with fitness maximization by default
       gen_alg1 = RealGA(fitness_function)
       gen_alg2 = RealGA(fitness_function)

       # initialize random one-dimensional populations of size 10 and 15 within interval (0, 1000)
       gen_alg1.init_random_population(10, 1, (0, 1000))
       gen_alg2.init_random_population(15, 1, (0, 1000))

       # then initialize migration GA using the already initialized standard GA instances
       mga = MigrationGA(type='real')  # set type of used instances
       mga.init_populations([gen_alg1, gen_alg2])

    Migration model with BinaryGA is used the same way. You may start computation by *mga.run(*args)*.
    """
    def __init__(self, type=u'binary'):
        u"""
        A constructor.

        Args:
            type (str): Type of used genetic algorithms: may be 'binary' or 'real'. Default is 'binary'.
        """
        self.type = type
        self._ga_list = None
        self._ga_list_size = None
        self._optim = None
        self._min_elements = numpy.inf

        self._check_parameters()

    def _check_parameters(self):
        if self.type not in [u'binary', u'real']:
            raise ValueError(u'Wrong value of input parameter.')

    def init_populations(self, ga_list):
        u"""
        This method initializes migration model of GA. Type of optimization ('min' or 'max')
        will be set to the same value of the first given GA instance. Valid GA instances are
        RealGA and BinaryGA.

        Args:
            ga_list (list): List of BinaryGA (or RealGA) instances with already initialized
                populations.
        """
        self._ga_list_size = len(ga_list)

        if self._ga_list_size < 2:
            raise ValueError(u'Too few populations.')

        for ga_inst in ga_list:
            if len(ga_inst.population) < self._min_elements:
                self._min_elements = len(ga_inst.population)

        self._ga_list = copy.deepcopy(ga_list)
        self._optim = ga_list[0].optim

    def _compare_solutions(self):
        u"""
        Compares best solutions of the specified GA instances and returns the best solution.

        Returns:
            best_solution (tuple): Best solution across all GA instances
            as (best chromosome, its fitness value).
        """
        if self._optim == u'min':
            # minimization
            best_solution = (None, numpy.inf)
            for ga_inst in self._ga_list:
                if ga_inst.best_solution[1] < best_solution[1]:
                    best_solution = ga_inst.best_solution
        else:
            # maximization
            best_solution = (None, -numpy.inf)
            for ga_inst in self._ga_list:
                if ga_inst.best_solution[1] > best_solution[1]:
                    best_solution = ga_inst.best_solution

        return best_solution

    def run(self, max_generation, period=1, migrant_num=1, cloning=True, migrate=True):
        u"""
        Runs a migration model of GA.

        Args:
            max_generation (int): Maximum number of GA generations.
            period (int): How often migration must be performed. Must be less than or equal to *max_generation*.
            migrant_num (int): How many best migrants will travel to all another populations.
            cloning (True, False): Can migrants clone? If False, an original population will not have
                its migrants after a migration. Otherwise, clones of migrants will remain
                in their original population after the migration of originals.
            migrate (True, False): Turns on/off migration process. It is useful in case of running GA by
                only *one* generation so *period* must be also set to 1, but you want to perform migration with period
                greater than 1 and thus, set migrate initially to False and set it to True when you actually want
                the algorithm to perform migration. This was used in benchmarking by COCO BBOB platform.

        Returns:
            fitness_progress, best_solution (tuple): *fitness_progress* contains lists of average fitness
            value of each generation for each specified GA instance. *best_solution* is the best solution
            across all GA instances as in form (best chromosome, its fitness value).

        You may use this method the standard way

        .. testsetup::

           from geneticalgs import RealGA, MigrationGA
           import math

           # define some function whose global minimum or maximum we are searching for
           # this function takes as input one-dimensional number
           def fitness_function(x):
               # the same function is used in examples
               return abs(x*(math.sin(x/11)/5 + math.sin(x/110)))

           # initialize two or more standard real GAs with fitness maximization by default
           gen_alg1 = RealGA(fitness_function)
           gen_alg2 = RealGA(fitness_function)

           # initialize random one-dimensional populations of size 10 and 15 within interval (0, 1000)
           gen_alg1.init_random_population(10, 1, (0, 1000))
           gen_alg2.init_random_population(15, 1, (0, 1000))

           # then initialize migration GA using the already initialized standard GA instances
           mga = MigrationGA(type='real')  # set type of used instances
           mga.init_populations([gen_alg1, gen_alg2])

        .. testcode::

           avg_fitness_progress, best_solution = mga.run(50, 10, 2)

        or in more unusual way if you want to get the best found solution for each generation

        .. testcode::

           max_generation = 10
           for i in range(max_generation):
               # perform migration every four generations
               if i > 0 and i % 3 == 0:
                   migrate = True
               else:
                   migrate = False

               _, best_solution = mga.run(1, 1, 2, cloning=True, migrate=migrate)

        """
        if max_generation < 1 or period > max_generation or period < 1 or\
                        migrant_num < 1 or migrant_num > self._min_elements or\
                        cloning not in [True, False] or migrate not in [True, False]:
            raise ValueError(u'Wrong value of the input parameter.')

        cycle = max_generation // period
        fitness_progress = [[] for i in xrange(self._ga_list_size)]

        for c in xrange(cycle):
            migrant_list = [[] for i in xrange(self._ga_list_size)]

            for ga_inst, index in izip(self._ga_list, xrange(self._ga_list_size)):
                # run standard GA and store average fitness progress
                fit_prog = ga_inst.run(period)

                if c < cycle - 1:
                    # the current fitness progress has the last value of the previous one
                    # we don't need the last fitness value twice
                    fitness_progress[index].extend(fit_prog[:-1])
                else:
                    fitness_progress[index].extend(fit_prog)

                if migrate:
                    for m in xrange(-migrant_num, 0, 1):
                        migrant_list[index].append(ga_inst.population[m])

                    if not cloning:
                        # no clones: remove the best *migrant_num* migrants
                        del ga_inst.population[-migrant_num:]

            # perform migration
            if migrate:
                for ga_inst, index in izip(self._ga_list, xrange(self._ga_list_size)):
                    # TODO uncomment in case of benchmarking using *my_experiment.py*
                    # del ga_inst.population[:migrant_num]  # uncomment for benchmarking on 2 populations
                    
                    for idx in xrange(self._ga_list_size):
                        if idx != index:
                            ga_inst.extend_population(migrant_list[idx])

        return fitness_progress, self._compare_solutions()





