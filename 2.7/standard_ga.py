from __future__ import division
from __future__ import absolute_import
import random
import numpy
from itertools import izip


class IndividualGA(object):
    u"""
    The class represents an individual of population in GA.
    """
    def __init__(self, individ, fitness_val):
        u"""
        A constructor.

        Args:
            individ (float, list): A chromosome represented a solution. The solution
                may be binary encoded in chromosome or be a float or a list of floats
                in case of dealing with real value solutions. The list contains
                only positions of bit 1 (according to self.data list) in case of binary encoded solution.
            fitness_val (int): Fitness value of the given chromosome.
        """
        self.individ = individ
        self.fitness_val = fitness_val


class StandardGA(object):
    u"""
    This class implements the base functionality of genetic algorithms and must be inherited.
    In other words, the class doesn't provide functionality of genetic algorithms by itself.
    This class is inherited by RealGA and BinaryGA classes in the current implementation.
    """
    def __init__(self, fitness_func=None, optim=u'max', selection=u"rank", mut_prob=0.05, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=True, tournament_size=None):
        u"""
        Args:
            fitness_func (function): This function must compute fitness value of a single individual.
                Function parameters must be: see subclasses.
            optim (str): What an algorithm must do with fitness value: maximize or minimize. May be 'min' or 'max'.
                Default is "max".
            selection (str): Parent selection type. May be "rank" (Rank Wheel Selection),
                "roulette" (Roulette Wheel Selection) or "tournament". Default is "rank".
            tournament_size (int): Defines the size of tournament in case of 'selection' == 'tournament'.
                Default is None.
            mut_prob (float): Probability of mutation. Recommended values are 0.5-1%. Default is 0.05.
            mut_type (int): This parameter defines how many random bits of individual are inverted (mutated).
                Default is 1.
            cross_prob (float): Probability of crossover. Recommended values are 80-95%. Default is 0.95.
            cross_type (int): This parameter defines crossover type. The following types are allowed:
                single point (1), two point (2) and multiple point (2 < cross_type <= len(data)).
                The extreme case of multiple point crossover is uniform one (cross_type == len(data)).
                The specified number of bits (cross_type) are crossed in case of multiple point crossover.
                Default is 1.
            elitism (True, False): Elitism on/off. Default is True.
        """
        self.fitness_func = fitness_func
        self.optim = optim
        self.selection = selection
        self.tournament_size = tournament_size
        self.mutation_prob = mut_prob
        self.mut_type = mut_type
        self.crossover_prob = cross_prob
        self.cross_type = cross_type
        self.elitism = elitism

        self._check_common_parameters()

        # population in standard model of GA
        self.population = None

        # mutation bit offset
        # default is 0
        self._mut_bit_offset = 0

        self.best_individ = None
        self.best_fitness = numpy.inf

    @property
    def best_solution(self):
        u"""
        Returns tuple in the following form: (best individual, best fitness value).

        Returns:
            tuple with the currently best found individual and its fitness value.
        """
        return self.best_individ, self.best_fitness

    def _check_common_parameters(self):
        u"""
        This function verifies common input parameters of a genetic algorithm.
        """
        if self.fitness_func is None or \
                self.optim not in [u'min', u'max'] or \
                self.mutation_prob < 0 or self.mutation_prob > 100 or \
                self.mut_type < 1 or \
                self.crossover_prob < 0 or self.crossover_prob > 100 or \
                self.cross_type < 1 or \
                self.selection not in [u"rank", u"roulette", u"tournament"] or \
                (self.selection == u'tournament' and self.tournament_size is None) or \
                self.elitism not in [True, False]:
            print u'Wrong value of input parameter.'
            raise ValueError

    def _random_diff(self, stop, n, start=0):
        u"""
        Creates a list of 'n' different random integer numbers within the interval (start, stop) ('start' included).

        Args:
            start (int): Start value of an interval (included). Default is 0.
            stop (int): End value of an interval (excluded).
            n (int): How many different random numbers must be generated.

        Returns:
             list of different random integer values from the given interval ('start' included)
        """
        if stop - start < n:
            # there is not enough numbers in the given interval
            print u'There is not enough numbers in the given interval'
            raise ValueError
        elif stop - start == n:
            # interval size == requested amount of numbers
            return xrange(start, stop)
        else:
            # requested amount of numbers is lower than the interval size
            random_number = random.randrange(start, stop)
            used_values = [random_number]

            for i in xrange(1, n):
                while random_number in used_values:
                    random_number = random.randrange(start, stop)

                used_values.append(random_number)

            return used_values

    def _invert_bit(self, individ, bit_num):
        u"""
        TO BE REIMPLEMENTED IN SUBCLASSES.
        This function mutates the appropriate bits from bit_num of the individual
        with the specified mutation probability.

        Args:
            individ (list): An individual of population.
            bit_num (list): List of bits' numbers to invert.

        Returns:
            mutated individual
        """
        raise NotImplementedError

    def _mutate(self, individ):
        u"""
        This function mutates (inverse bits) the given population individual.

        Args:
            individ (float, list): float or a list of floats, or a binary encoded combination
                of the original data list (it contains positions of bit 1 according to self.data).

        Returns:
             mutated individual as float, list of floats or binary representation (all with inverted bits)
        """
        if self._bin_length == self.mut_type:
            # it is necessary to mutate all bits with the specified mutation probability
            individ = self._invert_bit(individ, xrange(self._bin_length))
        else:
            # mutate some bits (not all)
            inverted_bits = self._random_diff(self._bin_length, self.mut_type, start=self._mut_bit_offset)
            individ = self._invert_bit(individ, inverted_bits)

        return individ

    def _replace_bits(self, source, target, start, stop):
        u"""
        TO BE REIMPLEMENTED IN SUBCLASSES.
        Replace target bits with source bits in interval (start, stop) (both included)
        with the specified crossover probability.

        Args:
            source (list): Values in source are used as replacement for target.
            target (list): Values in target are replaced with values in source.
            start (int): Start point of an interval (included).
            stop (int): End point of an interval (included).

        Returns:
             target with replaced bits with source one in the interval (start, stop) (both included)
        """
        raise NotImplementedError

    def _cross(self, parent1, parent2):
        u"""
        This function crosses the two given population individuals (parents).

        Args:
            parent1 (float, list): float or a list of floats, or a binary encoded combination
                of the original data list (self.data) of the first parent.
            parent2 (float, list): float or a list of floats, or a binary encoded combination
                of the original data list (self.data) of the second parent.

        Returns:
             list: an individual (binary representation, float or a list of floats) created by crossover of two parents
        """
        try:
            # a list of floats or binary encoded combination
            new_individ = list(parent1)
        except TypeError:
            # a single float
            new_individ = parent1

        if self.cross_type == self._bin_length:
            # it is necessary to replace all bits with the specified crossover probability
            for bit in xrange(self._bin_length):
                new_individ = self._replace_bits(parent2, new_individ, bit, bit)
        elif self.cross_type == 1:
            # combine two parts of parents
            random_bit = random.randrange(1, self._bin_length - 1)  # we want to do useful replacements
            new_individ = self._replace_bits(parent2, new_individ, random_bit + 1, self._bin_length - 1)
        elif self.cross_type == 2:
            # replace bits within  an interval of two random generated points
            random_bit1 = random.randrange(self._bin_length)  # we want to do useful replacements
            random_bit2 = random_bit1

            while random_bit2 == random_bit1:
                random_bit2 = random.randrange(self._bin_length)

            if random_bit1 < random_bit2:
                new_individ = self._replace_bits(parent2, new_individ, random_bit1, random_bit2)
            else:
                new_individ = self._replace_bits(parent2, new_individ, random_bit2, random_bit1)
        else:
            # cross some bits exactly (not replacement within an interval)
            cross_bits = self._random_diff(self._bin_length, self.cross_type)

            for bit in cross_bits:
                new_individ = self._replace_bits(parent2, new_individ, bit, bit)

        return new_individ

    def _conduct_tournament(self, population, size):
        u"""
        Conducts a tournament of the given size within the specified population.

        Args:
            population (list): All possible competitors. Population element is an IndividualGA object.
            size (int): Size of a tournament.

        Returns:
            winners (tuple of IndividualGA): winner of current tournament and the second best participant
        """
        population_size = len(population)

        if size > population_size or size < 1:
            print u'Wrong tournament size:', size
            raise ValueError

        if size == population_size:
            # sort by fitness value in the ascending order (maximization) or descending order (minimization)
            if self.optim == u'max':
                competitors = xrange(population_size)
            else:
                competitors = xrange(population_size - 1, 0, -1)
        else:
            competitors = self._random_diff(population_size, size)
            # sort by fitness value in the ascending order (maximization) or descending order (minimization)
            if self.optim == u'max':
                # ascending order (maximization)
                competitors.sort(key=lambda x: population[x].fitness_val)
            else:
                # descending order (minimization)
                competitors.sort(key=lambda x: population[x].fitness_val, reverse=True)

        # get the last two elements (winner and the second best participant)
        return competitors[-1], competitors[-2]

    def _select_parents(self, population, wheel_sum=None):
        u"""
        Selects parents from the given population (sorted in ascending or descending order).

        Args:
            population (list): Current population, sorted in ascending or descending order,
                from which parents will be selected. Population element is an IndividualGA object.
            wheel_sum (int): Sum of values on a wheel (different for "roulette" and "rank").

        Returns:
            parents (tuple of IndividualGA): selected parents
        """
        if self.selection in [u'roulette', u'rank']:
            if wheel_sum is None or wheel_sum < 0:
                print u'Wrong value of wheel sum:', wheel_sum
                raise ValueError

            parent1 = None
            parent2 = None
            wheel1 = random.uniform(0, wheel_sum)
            wheel2 = random.uniform(0, wheel_sum)

            sum_val = 0
            for ind, rank in izip(population, xrange(1, len(population) + 1)):
                if self.selection == u'roulette':
                    sum_val += ind.fitness_val
                else:
                    sum_val += rank

                if parent1 is None and sum_val > wheel1:
                    parent1 = ind
                if parent2 is None and sum_val > wheel2:
                    parent2 = ind

                if (parent1 is not None) and (parent2 is not None):
                    break

            return parent1, parent2
        elif self.selection == u'tournament':
            best1, second1 = self._conduct_tournament(population, self.tournament_size)
            best2, second2 = self._conduct_tournament(population, self.tournament_size)

            if population[best1].individ == population[best2].individ:
                return population[best1], population[second2]
            else:
                return population[best1], population[best2]
        else:
            print u'Unknown selection type:', self.selection
            raise ValueError

    def _sort_population(self):
        u"""
        Sorts self.population according to self.optim ("min" or "max") in such way
        that the last element of the population in both cases is an individual with the best fitness value.
        """
        if self.optim == u'max':
            # an algorithm maximizes a fitness value
            # ascending order
            self.population.sort(key=lambda x: x.fitness_val)
        else:
            # an algorithm minimizes a fitness value
            # descending order
            self.population.sort(key=lambda x: x.fitness_val, reverse=True)

    def _update_solution(self, individ, fitness_val):
        u"""
        Updates current best solution if the given one is better.

        Args:
            individ (float, list): Individual of a population (binary encoded, float or list of floats).
            fitness_val (float, int): Fitness value of the given individual.
        """
        if fitness_val < self.best_fitness:
            self.best_individ = individ
            self.best_fitness = fitness_val

    def _compute_rank_wheel_sum(self, population_size):
        u"""
        The function returns sum of a wheel that is necessary in parent selection process
        in case of "rank" selection type.

        Args:
            population_size (int): Size of a population.

        Returns:
            sum of a wheel for the given population size
        """
        return numpy.cumsum(xrange(1, population_size + 1))[-1]

    def _compute_fitness(self, individ):
        u"""
        TO BE REIMPLEMENTED IN SUBCLASSES.
        This function computes fitness value of the given individual.

        Args:
            individ (float, list): An individual of genetic algorithm.
                Defined fitness function (self.fitness_func) must deal with this individual.

        Returns:
            fitness value of the given individual
        """
        raise NotImplementedError

    def _check_init_random_population(self, *args):
        u"""
        TO BE REIMPLEMENTED IN SUBCLASSES.

        This function verifies the input parameters of a random initialization.
        """
        raise NotImplementedError

    def _generate_random_population(self, *args):
        u"""
        TO BE REIMPLEMENTED IN SUBCLASSES.

        This function generates new random population by the given input parameters.
        """
        raise NotImplementedError

    def init_population(self, new_population):
        u"""
        Initializes population with the given individuals (individual as binary encoded, float or a list of floats)
        of 'new_population'. The fitness values of these individuals will be computed by a specified fitness function.

        It is recommended to have new_population size equal to some squared number (9, 16, 100, 625 etc.)
        in case of diffusion model of GA. Otherwise some last individuals in the given population will be lost
        as the current implementation works only with square arrays of diffusion model.

        Args:
            new_population (list): New initial population of individuals. A single individual in case of binary GA
                is represented as a list of bits' positions with value 1 in the following way:
                LSB (least significant bit) has position (len(self.data) - 1) and
                MSB (most significant bit) has position 0. If it is a GA on real values, an individual is represented
                as a float or a list of floats in case of multiple dimensions.
        """
        if not new_population or len(new_population) < 4:
            print u'New population is too few.'
            raise ValueError

        self.population = []
        for individ in new_population:
            fit_val = self._compute_fitness(individ)
            self.population.append(IndividualGA(individ, fit_val))

        self._sort_population()
        self._update_solution(self.population[-1].individ, self.population[-1].fitness_val)

    def extend_population(self, elem_list):
        u"""
        Extends a current population with new elements. Be careful with type of elements
        in *elem_list*: they must have the same type as elements
        of a current population (IndividualGA for BinaryGA, a float or a list of floats for RealGA).

        Args:
            elem_list (list): New elements of the same type as in the population to be extended.
        """
        self.population.extend(elem_list)

        self._sort_population()
        self._update_solution(self.population[-1].individ, self.population[-1].fitness_val)

    def run(self, max_generation):
        u"""
        Starts a standard GA. The algorithm does 'max_generation' generations and then stops.
        Old population is completely replaced with new one.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            list of average fitness values for each generation (including original population)
        """
        if max_generation < 1:
            print u'Too few generations...'
            raise ValueError

        fitness_progress = []
        fitness_sum = -1
        population_size = None

        for generation_num in xrange(max_generation):
            fitness_sum = sum(ind.fitness_val for ind in self.population)
            population_size = len(self.population)
            next_population = []
            fitness_progress.append(fitness_sum / population_size)

            for i in xrange(population_size):
                if self.selection == u'roulette':
                    parent1, parent2 = self._select_parents(self.population, fitness_sum)
                elif self.selection == u'rank':
                    parent1, parent2 = self._select_parents(self.population,
                                                            self._compute_rank_wheel_sum(population_size)
                                                            )
                else:
                    # tournament
                    parent1, parent2 = self._select_parents(self.population)

                # cross parents and mutate a child
                new_individ = self._mutate(self._cross(parent1.individ, parent2.individ))
                # compute fitness value of the child
                fit_val = self._compute_fitness(new_individ)

                next_population.append(IndividualGA(new_individ, fit_val))

            if self.elitism:
                # copy the best individual to a new generation
                next_population.append(self.population[-1])

            self.population = next_population
            self._sort_population()
            self._update_solution(self.population[-1].individ, self.population[-1].fitness_val)

        fitness_progress.append(fitness_sum / population_size)

        return fitness_progress

