import numpy
import copy


class MigrationGA:
    """
    This class implements migration model of GA, namely island model (not stepping-stone).
    It can work with binary or real GA.
    """
    def __init__(self, type='binary'):
        """
        A constructor.

        Args:
            type (str): Type of genetic algorithm: may be 'binary' or 'real'.
        """
        self.type = type
        self.population_list = None
        self._population_size = None
        self._optim = None

        self._check_parameters()

    def _check_parameters(self):
        if self.type not in ['binary', 'real']:
            print('Wrong value of input parameter.')
            raise ValueError

    def init_populations(self, population_list):
        # TODO doctest examples
        """
        This function initializes migration model of GA.

        Args:
            population_list (list): List of BinaryGA (or RealGA) instances with already initialized
                populations.
        """
        self._population_size = len(population_list)

        if self._population_size < 2:
            print('Too few populations.')
            raise ValueError

        self.population_list = copy.deepcopy(population_list)
        self._optim = population_list[0].optim

    def _compare_solutions(self):
        """
        Compares best solutions of the specified GA instances and returns the best solution.

        Returns:
            best_solution (tuple): Best solution across all GA instances
                as (best individual, best fitness value).
        """
        if self._optim == 'min':
            # minimization
            best_solution = (None, numpy.inf)
            for ga_inst in self.population_list:
                if ga_inst.best_solution[1] < best_solution[1]:
                    best_solution = ga_inst.best_solution
        else:
            # maximization
            best_solution = (None, -numpy.inf)
            for ga_inst in self.population_list:
                if ga_inst.best_solution[1] > best_solution[1]:
                    best_solution = ga_inst.best_solution

        return best_solution

    def run(self, max_generation, period=1, migrant_num=1, cloning=True, migrate=True):
        """
        Runs a migration model of GA.

        Args:
            max_generation (int): Maximum number of GA generations.
            period (int): How often performs migration. Must be less than *max_generation*.
            migrant_num (int): How many best migrants will travel to all another populations.
            cloning (True, False): Can migrants clone? If False, an original population will not have
                its migrants after a migration. Otherwise, clones of migrants will remain
                in their original population after the migration of originals.

        Returns:
             fitness_progress, best_solution (tuple): *fitness_progress* contains lists of average fitness
                value of each generation for each specified GA instance. *best_solution* is the best solution
                across all GA instances as in form (best individual, best fitness value).
        """
        if max_generation < 1 or period > max_generation or \
                        migrant_num < 1 or cloning not in [True, False]:
            print('Wrong value of input parameter.')
            raise ValueError

        cycle = max_generation // period
        fitness_progress = [[] for i in range(self._population_size)]

        for c in range(cycle):
            migrant_list = [[] for i in range(self._population_size)]

            for ga_inst, index in zip(self.population_list, range(self._population_size)):
                # run standard GA and store average fitness progress
                fit_prog = ga_inst.run(period)
                fitness_progress[index].extend(fit_prog)

		if migrate:
                    for m in range(-migrant_num, 0, 1):
                        migrant_list[index].append(ga_inst.population[m])

                    if not cloning:
                        # no clones: remove the best *migrant_num* migrants
                        del ga_inst.population[-migrant_num:]

            # perform migration
	    if migrate:
                for ga_inst, index in zip(self.population_list, range(self._population_size)):
		    # TODO
		    # del ga_inst.population[:migrant_num]  # uncomment for benchmarking on 2 populations

                    for idx in range(self._population_size):
                        if idx != index:
                            ga_inst.extend_population(migrant_list[idx])

        return fitness_progress, self._compare_solutions()





