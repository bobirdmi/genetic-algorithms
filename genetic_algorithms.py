import random
import numpy


class IndividualGA:
    """
    The class represents an individual of population in GA.
    """
    def __init__(self, individ, fitness_val):
        """
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


class GeneticAlgorithms:
    def __init__(self, fitness_func=None, optim='max', type='standard', selection="rank", mut_prob=0.05, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=True, tournament_size=None):
        """
        Args:
            fitness_func (function): This function must compute fitness value of a single individual.
                Function parameters must be: see subclasses.
            optim (str): What an algorithm must do with fitness value: maximize or minimize. May be 'min' or 'max'.
                Default is "max".
            type (str): Type of genetic algorithm. May be 'standard', 'diffusion' or 'migration'.
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
        self.type = type
        self.selection = selection
        self.tournament_size = tournament_size
        self.mutation_prob = mut_prob
        self.mut_type = mut_type
        self.crossover_prob = cross_prob
        self.cross_type = cross_type
        self.elitism = elitism
        self.population = None

        self._check_common_parameters()

        # mutation bit offset
        # default is 0
        self._mut_bit_offset = 0

    def _check_common_parameters(self):
        if self.fitness_func is None or \
                self.optim not in ['min', 'max'] or \
                self.type not in ['standard', 'diffusion', 'migration'] or \
                self.mutation_prob < 0 or self.mutation_prob > 100 or \
                self.mut_type < 1 or \
                self.crossover_prob < 0 or self.crossover_prob > 100 or \
                self.cross_type < 1 or \
                self.selection not in ["rank", "roulette", "tournament"] or \
                (self.selection == 'tournament' and self.tournament_size is None) or \
                self.elitism not in [True, False]:
            print('Wrong value of input parameter.')
            raise ValueError

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
            # there is not enough numbers in the given interval
            print('There is not enough numbers in the given interval')
            raise ValueError
        elif stop - start == n:
            # interval size == requested amount of numbers
            return list(range(start, stop))
        else:
            # requested amount of numbers is lower than the interval size
            random_number = random.randrange(start, stop)
            used_values = [random_number]

            for i in range(1, n):
                while random_number in used_values:
                    random_number = random.randrange(start, stop)

                used_values.append(random_number)

            return used_values

    def _invert_bit(self, individ, bit_num):
        """
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
        """
        This function mutates (inverse bits) the given population individual.

        Args:
            individ (float, list): float or a list of floats, or a binary encoded combination
                of the original data list (it contains positions of bit 1 according to self.data).

        Returns:
             mutated individual as float, list of floats or binary representation (all with inverted bits)
        """
        if self._bin_length == self.mut_type:
            # it is necessary to mutate all bits with the specified mutation probability
            individ = self._invert_bit(individ, list(range(self._bin_length)))
        else:
            # mutate some bits (not all)
            inverted_bits = self._random_diff(self._bin_length, self.mut_type, start=self._mut_bit_offset)
            individ = self._invert_bit(individ, inverted_bits)

        return individ

    def _replace_bits(self, source, target, start, stop):
        """
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
        """
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
            for bit in range(self._bin_length):
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
        """
        Conducts a tournament of the given size within the specified population.

        Args:
            population (list): All possible competitors. Population element is an IndividualGA object.
            size (int): Size of a tournament.

        Returns:
            winners (tuple of IndividualGA): winner of current tournament and the second best participant
        """
        population_size = len(population)

        if size > population_size or size < 1:
            print('Wrong tournament size:', size)
            raise ValueError

        if size == population_size:
            # sort by fitness value in the ascending order (maximization) or descending order (minimization)
            if self.optim == 'max':
                competitors = list(range(population_size))
            else:
                competitors = list(range(population_size - 1, 0, -1))
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

    def _get_neighbour_coords(self, loc, shape):
        """
        This function randomly selects neighbour of the given individual in diffusion array
        and returns coordinates of this neighbour.

        Args:
            loc (tuple): Coordinates of a current individual as (row, column).
            shape (tuple): Shape of an array of all population individuals (row, column). The array elements are
                IndividualGA objects.

        Returns:
            coordinates (tuple): coordinates of a randomly selected neighbour as (row, column).
        """
        up, down, left, right = (0, 1, 2, 3)
        DIRS = {
            up: ((loc[0] - 1) % shape[0], loc[1]),
            down: ((loc[0] + 1) % shape[0], loc[1]),
            left: (loc[0], (loc[1] - 1) % shape[1]),
            right: (loc[0], (loc[1] + 1) % shape[1])
        }

        return DIRS[random.randrange(4)]

    def _compute_diffusion_generation(self, array):
        """
        This function computes a new generation of the diffusion model of GA.

        Args:
            array (array of IndividualGA): Diffusion array of the current generation.

        Returns:
            new_array (array of IndividualGA): Computed diffusion array of the next generation.
        """
        shape = array.shape
        new_array = numpy.zeros(shape)

        for row in range(shape[0]):
            for column in range(shape[1]):
                neighbour_coords = self._get_neighbour_coords((row, column), shape)
                parent1 = array[row, column]
                parent2 = array[neighbour_coords]

                # cross parents and mutate a child
                new_individ = self._mutate(self._cross(parent1.individ, parent2.individ))
                # compute fitness value of the child
                fit_val = self._compute_fitness(new_individ)

                new_array[row, column] = IndividualGA(new_individ, fit_val)

        return new_array

    def _select_parents(self, population, wheel_sum=None):
        """
        Selects parents from the given population (sorted in ascending or descending order).

        Args:
            population (list): Current population, sorted in ascending or descending order,
                from which parents will be selected. Population element is an IndividualGA object.
            wheel_sum (int): Sum of values on a wheel (different for "roulette" and "rank").

        Returns:
            parents (tuple of IndividualGA): selected parents
        """
        if self.selection in ['roulette', 'rank']:
            if wheel_sum is None or wheel_sum < 2:
                print('Wrong value of wheel sum:', wheel_sum)
                raise ValueError

            parent1 = None
            parent2 = None
            wheel1 = random.uniform(0, wheel_sum)
            wheel2 = random.uniform(0, wheel_sum)

            sum_val = 0
            for ind, rank in zip(population, range(1, len(population) + 1)):
                # population is sorted in ascending or descending order by fitness values
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

            if population[best1].individ == population[best2].individ:
                return population[best1], population[second2]
            else:
                return population[best1], population[best2]
        else:
            print('Unknown selection type:', self.selection)
            raise ValueError

    def _sort_population(self):
        if self.optim == 'max':
            # an algorithm maximizes a fitness value
            # ascending order
            self.population.sort(key=lambda x: x.fitness_val)
        else:
            # an algorithm minimizes a fitness value
            # descending order
            self.population.sort(key=lambda x: x.fitness_val, reverse=True)

    def _compute_fitness(self, individ):
        """
        TO BE REIMPLEMENTED IN SUBCLASSES.
        This function computes fitness value of the given individual.

        Args:
            individ (float, list): An individual of genetic algorithm.
                Defined fitness function (self.fitness_func) must deal with this individual.

        Returns:
            fitness value of the given individual
        """
        raise NotImplementedError

    def init_population(self, new_population):
        """
        Initializes population with the given binary encoded individuals of 'new_population'. The fitness values 
        of these individuals will be computed by a specified fitness function.

        Args:
            new_population (list): New initial population of binary encoded individuals. A single individual is represented
                as a list of bits' positions with value 1 in the following way: LSB (least significant bit)
                has position (len(self.data) - 1) and MSB (most significant bit) has position 0.
        """
        if not new_population:
            print('New population is empty')
            raise ValueError

        # TODO diffusion model
        self.population = []
        for individ in new_population:
            fit_val = self._compute_fitness(individ)
            self.population.append(IndividualGA(individ, fit_val))

        self._sort_population()

    def _run_standard(self, max_generation):
        """
        Starts a standard GA. The algorithm does 'max_generation' generations and then stops.
        Old population is completely replaced with new one.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            list of average fitness values for each generation (including original population)
        """
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
                                                            numpy.cumsum(range(1, population_size + 1))[-1]
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

        fitness_progress.append(fitness_sum / population_size)

        return fitness_progress

    def _run_diffusion(self, max_generation):
        """
        Starts a diffusion GA. The algorithm does 'max_generation' generations and then stops.
        Old population is completely replaced with new one.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            list of average fitness values for each generation (including original population)
        """
        # TODO
        fitness_progress = []
        # self.population is numpy array in case of diffusion model
        population_size = self.population.size
        shape = self.population.shape

        for generation_num in range(max_generation):
            fitness_sum = 0
            for row in range(shape[0]):
                for column in range(shape[1]):
                    fitness_sum += self.population[row, column].fitness_val

            next_population = []
            fitness_progress.append(fitness_sum / population_size)

            if self.elitism:
                # copy the best individual to a new generation
                next_population.append(self.population[-1])

            self.population = next_population

        fitness_progress.append(fitness_sum / population_size)

        return fitness_progress

    def run(self, max_generation):
        """
        Starts GA. The algorithm does 'max_generation' generations and then stops.
        Old population is completely replaced with new one.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            list of average fitness values for each generation (including original population)
        """
        if max_generation < 1:
            print('Too few generations...')
            raise ValueError

        if self.type == 'diffusion':
            fitness_progress = self._run_diffusion(max_generation)
        elif self.type == 'standard':
            fitness_progress = self._run_standard(max_generation)
        else:
            # migration model of GA
            # TODO
            fitness_progress = []

        return fitness_progress

