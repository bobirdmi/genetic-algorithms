import random
import numpy


class IndividualGA:
    """
    The class represents an individual of population in GA.
    """
    def __init__(self, chromosome, fitness_val):
        """
        A constructor.

        Args:
            chromosome (float, list): A chromosome represented a solution. The solution
                may be binary encoded in chromosome or be a float or a list of floats
                in case of dealing with real value solutions. The list contains
                only positions of bit 1 (according to self.data list) in case of binary encoded solution.
            fitness_val (float, int): Fitness value of the given chromosome.
        """
        self.chromosome = chromosome
        self.fitness_val = fitness_val


class StandardGA:
    """
    This class implements the base functionality of genetic algorithms and must be inherited.
    In other words, the class doesn't provide functionality of genetic algorithms by itself.
    This class is inherited by RealGA and BinaryGA classes in the current implementation.
    """
    def __init__(self, fitness_func=None, optim='max', selection="rank", mut_prob=0.05, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=True, tournament_size=None):
        """
        Args:
            fitness_func (function): This function must compute fitness value of a single chromosome.
                Function parameters depend on the implemented subclasses of this class.
            optim (str): What this genetic algorithm must do with fitness value: maximize or minimize.
                May be 'min' or 'max'. Default is "max".
            selection (str): Parent selection type. May be "rank" (Rank Wheel Selection),
                "roulette" (Roulette Wheel Selection) or "tournament". Default is "rank".
            tournament_size (int): Defines the size of tournament in case of 'selection' == 'tournament'.
                Default is None.
            mut_prob (float): Probability of mutation. Recommended values are 0.5-1%. Default is 0.5% (0.05).
            mut_type (int): This parameter defines how many chromosome bits will be mutated. Default is 1.
            cross_prob (float): Probability of crossover. Recommended values are 80-95%. Default is 95% (0.95).
            cross_type (int): This parameter defines crossover type. The following types are allowed:
                single point (1), two point (2) and multiple point (2 < *cross_type*).
                The extreme case of multiple point crossover is uniform one (*cross_type* == all_bits).
                The specified number of bits (*cross_type*) are crossed in case of multiple point crossover.
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

        self.best_chromosome = None
        if optim == 'min':
            self.best_fitness = numpy.inf
        else:
            self.best_fitness = -numpy.inf

    @property
    def best_solution(self):
        """
        Returns tuple in the following form: (best chromosome, its fitness value).

        Returns:
            tuple with the currently best found chromosome and its fitness value.
        """
        return self.best_chromosome, self.best_fitness

    def _check_common_parameters(self):
        """
        This function verifies common input parameters of a genetic algorithm.
        """
        if self.fitness_func is None or \
                self.optim not in ['min', 'max'] or \
                self.mutation_prob < 0 or self.mutation_prob > 1 or \
                self.mut_type < 1 or \
                self.crossover_prob < 0 or self.crossover_prob > 1 or \
                self.cross_type < 1 or \
                self.selection not in ["rank", "roulette", "tournament"] or \
                (self.selection == 'tournament' and self.tournament_size is None) or \
                self.elitism not in [True, False]:
            raise ValueError('Wrong value of input parameter.')

    def _random_diff(self, stop, n, start=0):
        """
        Creates a list of 'n' different random integer numbers within the interval (start, stop) ('start' included).

        Args:
            start (int): Start value of an interval (included). Default is 0.
            stop (int): End value of an interval (excluded).
            n (int): How many different random numbers must be generated.

        Returns:
             list of different random integer values from the given interval ('start' included)
        """
        if stop - start < n:
            # there are not enough numbers in the given interval
            raise ValueError('There is not enough numbers in the given interval.')
        elif stop - start == n:
            # interval size == requested amount of numbers
            return list(range(start, stop))
        else:
            # requested amount of numbers is less than the interval size
            random_number = random.randrange(start, stop)
            used_values = [random_number]

            for i in range(1, n):
                while random_number in used_values:
                    random_number = random.randrange(start, stop)

                used_values.append(random_number)

            return used_values

    def _invert_bit(self, chromosome, bit_num):
        """
        TO BE REIMPLEMENTED IN SUBCLASSES.
        This function mutates the appropriate bits of the chromosome from *bit_num*
        with the specified mutation probability.

        Args:
            chromosome (list, float): A chromosome of population (chromosome without its fitness value).
            bit_num (list): List of bits' numbers to invert.

        Returns:
            mutated chromosome
        """
        raise NotImplementedError('This function must be reimplemented in subclasses.')

    def _mutate(self, chromosome):
        """
        This function mutates (inverses bits) the given chromosome.

        Args:
            chromosome (float, list): a float or a list of floats, or a binary encoded combination
                of the original data list (it contains positions of bit 1 according to *self.data*).

        Returns:
             mutated chromosome as float, list of floats or binary representation (any of the mentioned
                representations with inverted bits depending on subclass)
        """
        if self._bin_length == self.mut_type:
            # it is necessary to mutate all bits with the specified mutation probability
            chromosome = self._invert_bit(chromosome, list(range(self._bin_length)))
        else:
            # mutate some bits (not all)
            inverted_bits = self._random_diff(self._bin_length, self.mut_type, start=self._mut_bit_offset)
            chromosome = self._invert_bit(chromosome, inverted_bits)

        return chromosome

    def _replace_bits(self, source, target, start, stop):
        """
        TO BE REIMPLEMENTED IN SUBCLASSES.
        Replace target bits with source bits in interval (start, stop) (both included)
        with the specified crossover probability. This interval represents
        positions of bits to replace (minimum start point is 0 and maximum end point is *self._bin_length - 1*).

        Args:
            source (list): Values in source are used as replacement for target.
            target (list): Values in target are replaced with values in source.
            start (int): Start point of an interval (included).
            stop (int): End point of an interval (included).

        Returns:
             target with replaced bits with source one in the interval (start, stop) (both included)
        """
        raise NotImplementedError('This function must be reimplemented in subclasses.')

    def _cross(self, parent1, parent2):
        """
        This function crosses over the two given chromosomes (parents). The first parent is a target chromosome
        that means its bits will be replaced with bits of the second parent (source chromosome) with
        the specified crossover probability.

        Args:
            parent1 (float, list): Target chromosome. May be a float or a list of floats, or a binary encoded combination
                of the original data list (*self.data*) of the first parent.
            parent2 (float, list): Source chromosome. May be a float or a list of floats, or a binary encoded combination
                of the original data list (*self.data*) of the second parent.

        Returns:
             child (list, float): a chromosome (a binary representation, a float or a list of floats) created by the
                crossover of the two given parents
        """
        try:
            # a list of floats or binary encoded combination
            new_chromosome = list(parent1)
        except TypeError:
            # a single float
            new_chromosome = parent1

        if self.cross_type == self._bin_length:
            # it is necessary to replace all bits with the specified crossover probability
            new_chromosome = self._replace_bits(parent2, new_chromosome, 0, self._bin_length - 1)
        elif self.cross_type == 1:
            # combine two parts of parents
            random_bit = random.randrange(1, self._bin_length - 1)  # we want to do useful replacements
            new_chromosome = self._replace_bits(parent2, new_chromosome, random_bit + 1, self._bin_length - 1)
        elif self.cross_type == 2:
            # replace bits within  an interval of two random generated points
            random_bit1 = random.randrange(self._bin_length)  # we want to do useful replacements
            random_bit2 = random_bit1

            while random_bit2 == random_bit1:
                random_bit2 = random.randrange(self._bin_length)

            if random_bit1 < random_bit2:
                new_chromosome = self._replace_bits(parent2, new_chromosome, random_bit1, random_bit2)
            else:
                new_chromosome = self._replace_bits(parent2, new_chromosome, random_bit2, random_bit1)
        else:
            # cross some bits exactly (not replacement within an interval)
            cross_bits = self._random_diff(self._bin_length, self.cross_type)

            for bit in cross_bits:
                new_chromosome = self._replace_bits(parent2, new_chromosome, bit, bit)

        return new_chromosome

    def _conduct_tournament(self, population, size):
        """
        Conducts a tournament of the given size within the specified population. The population must be
        sorted by chromosome's fitness value the following way: the last population elements are the best.

        Args:
            population (list): All possible competitors. Size of the population must be at least 2.
                Population element is an IndividualGA object.
            size (int): Size of a tournament. It will be set to the whole population,
                if it is greater than the given population size.

        Returns:
            winners (int, int): indices of a winner of the current tournament and the second best participant
        """
        if size < 1 or population is None:
            raise ValueError('Wrong input parameter.')

        try:
            population_size = len(population)
        except TypeError:
            raise ValueError('Population must be a list.')

        if population_size < 1:
            raise ValueError('Too small population.')

        if size > population_size:
            size = population_size

        if size == population_size:
            # the population is already sorted and tournament is conducted across the whole population
            competitors = list(range(population_size))
        else:
            competitors = self._random_diff(population_size, size)
            # sort by fitness value in the ascending order (maximization) or descending order (minimization)
            if self.optim == 'max':
                # ascending order (maximization)
                competitors.sort(key=lambda x: population[x].fitness_val)
            else:
                # descending order (minimization)
                competitors.sort(key=lambda x: population[x].fitness_val, reverse=True)

        # get the last two elements (winner and the second best participant)
        return competitors[-1], competitors[-2]

    def _select_parents(self, population, wheel_sum=None):
        """
        Selects parents from the given population.

        Args:
            population (list): Current population from which parents will be selected.
                Population element is an IndividualGA object.
            wheel_sum (float): Sum of values on a wheel (different for "roulette" and "rank").

        Returns:
            parents (IndividualGA, IndividualGA): selected parents
        """
        if self.selection in ['roulette', 'rank']:
            if wheel_sum is None or wheel_sum <= 0:
                print('Wrong value of wheel sum:', wheel_sum)
                raise ValueError('Wrong value of wheel sum')

            parent1 = None
            parent2 = None
            wheel1 = random.uniform(0, wheel_sum)
            wheel2 = random.uniform(0, wheel_sum)

            sum_val = 0
            for ind, rank in zip(population, range(1, len(population) + 1)):
                if self.selection == 'roulette':
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
        elif self.selection == 'tournament':
            best1, second1 = self._conduct_tournament(population, self.tournament_size)
            best2, second2 = self._conduct_tournament(population, self.tournament_size)

            if population[best1].chromosome == population[best2].chromosome:
                return population[best1], population[second2]
            else:
                return population[best1], population[best2]
        else:
            print('Unknown selection type:', self.selection)
            raise ValueError('Unknown selection type')

    def _sort_population(self):
        """
        Sorts self.population according to *self.optim* ("min" or "max") in such way
        that the last element of the population in both cases is the chromosome with the best fitness value.
        """
        if self.optim == 'max':
            # an algorithm maximizes a fitness value
            # ascending order
            self.population.sort(key=lambda x: x.fitness_val)
        else:
            # an algorithm minimizes a fitness value
            # descending order
            self.population.sort(key=lambda x: x.fitness_val, reverse=True)

    def _update_solution(self, chromosome, fitness_val):
        """
        Updates current best solution if the given one is better.

        Args:
            chromosome (float, list): Chromosome of a population (binary encoded, float or list of floats).
            fitness_val (float, int): Fitness value of the given chromosome.
        """
        if (self.optim == 'min' and fitness_val < self.best_fitness)\
                or (self.optim == 'max' and fitness_val > self.best_fitness):
            self.best_chromosome = chromosome
            self.best_fitness = fitness_val

    def _compute_rank_wheel_sum(self, population_size):
        """
        The function returns sum of a wheel that is necessary in parent selection process
        in case of "rank" selection type.

        Args:
            population_size (int): Size of a population.

        Returns:
            sum of the wheel for the given population size
        """
        return numpy.cumsum(range(1, population_size + 1))[-1]

    def _compute_fitness(self, chromosome):
        """
        TO BE REIMPLEMENTED IN SUBCLASSES.
        This function computes fitness value of the given chromosome.

        Args:
            chromosome (float, list): A chromosome of genetic algorithm.
                Defined fitness function (self.fitness_func) must deal with such chromosomes.

        Returns:
            fitness value of the given chromosome
        """
        raise NotImplementedError('This function must be reimplemented in subclasses.')

    def _check_init_random_population(self, *args):
        """
        TO BE REIMPLEMENTED IN SUBCLASSES.

        This function verifies the input parameters of a random initialization.
        """
        raise NotImplementedError('This function must be reimplemented in subclasses.')

    def _generate_random_population(self, *args):
        """
        TO BE REIMPLEMENTED IN SUBCLASSES.

        This function generates new random population by the given input parameters.
        """
        raise NotImplementedError('This function must be reimplemented in subclasses.')

    def init_population(self, chromosomes, interval=None):
        """
        Initializes a population with the given chromosomes (binary encoded, float or a list of floats).
        The fitness values of these chromosomes will be computed by a specified fitness function.

        It is recommended to have an amount of chromosomes equal to some squared number (9, 16, 100, 625 etc.)
        in case of diffusion model of GA. Otherwise some last chromosomes will be lost
        as the current implementation supports only square arrays of diffusion model.

        Args:
            chromosomes (list): Chromosomes of a new population. A single chromosome in case of binary GA
                is represented as a list of bits' positions with value 1 in the following way:
                LSB (least significant bit) has position (len(self.data) - 1) and
                MSB (most significant bit) has position 0. If it is a GA on real values, a chromosome is represented
                as a float or a list of floats in case of multiple dimensions. Size of *chromosomes* list must be
                at least 4.
            interval (tuple): An interval in which we are searching the best solution.
                Must be specified in case of RealGA.
        """
        if not chromosomes or len(chromosomes) < 4:
            raise ValueError('New population is too small.')

        if not hasattr(self, '_data'):
            if interval is None or interval[0] >= interval[1]:
                raise ValueError('You must specify a correct interval for RealGA.')

            self.interval = interval

        self.population = []
        for chromosome in chromosomes:
            fit_val = self._compute_fitness(chromosome)
            self.population.append(IndividualGA(chromosome, fit_val))

        self._sort_population()
        self._update_solution(self.population[-1].chromosome, self.population[-1].fitness_val)

    def extend_population(self, elem_list):
        """
        DOES NOT WORK WITH DIFFUSION GENETIC ALGORITHM.

        Extends a current population with the new elements. Be careful with type of elements
        in *elem_list*: they must have the same type as elements of a current population,
        e.g. IndividualGA objects with the *appropriate* chromosome representation
        (binary encoded for BinaryGA, a float or a list of floats for RealGA).

        Args:
            elem_list (list): New elements of the same type (including chromosome representation)
                as in the current population.
        """
        self.population.extend(elem_list)

        self._sort_population()
        self._update_solution(self.population[-1].chromosome, self.population[-1].fitness_val)

    def run(self, max_generation):
        """
        Starts a standard GA (RealGA or BinaryGA). The algorithm performs *max_generation* generations and then stops.
        Old population is completely replaced with a new computed one at the end of each generation.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            fitness_progress (list): List of average fitness values for each generation (including original population)
        """
        if max_generation < 1:
            raise ValueError('Too few generations...')

        fitness_progress = []
        fitness_sum = -1
        population_size = None

        for generation_num in range(max_generation):
            fitness_sum = sum(ind.fitness_val for ind in self.population)
            population_size = len(self.population)
            next_population = []
            fitness_progress.append(fitness_sum / population_size)

            for i in range(population_size):
                if self.selection == 'roulette':
                    parent1, parent2 = self._select_parents(self.population, fitness_sum)
                elif self.selection == 'rank':
                    parent1, parent2 = self._select_parents(self.population,
                                                            self._compute_rank_wheel_sum(population_size)
                                                            )
                else:
                    # tournament
                    parent1, parent2 = self._select_parents(self.population)

                # cross parents and mutate a child
                new_chromosome = self._mutate(self._cross(parent1.chromosome, parent2.chromosome))
                # compute fitness value of the child
                fit_val = self._compute_fitness(new_chromosome)

                next_population.append(IndividualGA(new_chromosome, fit_val))

            if self.elitism:
                # copy the best individual to a new generation
                next_population.append(self.population[-1])

            self.population = next_population
            self._sort_population()
            self._update_solution(self.population[-1].chromosome, self.population[-1].fitness_val)

        fitness_progress.append(fitness_sum / population_size)

        return fitness_progress

