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
            individ (list): A chromosome represented a solution. The solution is binary encoded in chromosome.
                This list contains only positions of bit 1 (according to self.data list).
            fitness_val (int): Fitness value of the given chromosome.
        """
        self.individ = list(individ)
        self.fitness_val = fitness_val


class GeneticAlgorithms:
    def __init__(self, data=None, fitness_func=None, optim='max', selection="rank", mut_prob=0.5, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=False, tournament_size=None):
        """
        Args:
            data (list): A list with elements whose combination will be binary encoded and
                evaluated by a fitness function. Minimum amount of elements is 4.
            fitness_func (function): This function must compute fitness value of an input combination of the given data.
                Function parameters must be: list of used indices of the given data (from 0), list of data itself.
            optim (str): What an algorithm must do with fitness value: maximize or minimize. May be 'min' or 'max'.
            selection (str): Parent selection type. May be "rank" (Rank Wheel Selection),
                "roulette" (Roulette Wheel Selection) or "tournament".
            tournament_size (int): Defines the size of tournament in case of 'selection' == 'tournament'.
            mut_prob (float): Probability of mutation. Recommended values are 0.5-1%.
            mut_type (int): This parameter defines how many random bits of individual are inverted (mutated).
            cross_prob (float): Probability of crossover. Recommended values are 80-95%.
            cross_type (int): This parameter defines crossover type. The following types are allowed:
                single point (1), two point (2) and multiple point (2 < cross_type <= len(data)).
                The extreme case of multiple point crossover is uniform one (cross_type == len(data)).
                The specified number of bits (cross_type) are crossed in case of multiple point crossover.
            elitism (True, False): Elitism on/off.
        """
        self.data = data
        self.fitness_func = fitness_func
        self.optim = optim
        self.selection = selection
        self.tournament_size = tournament_size
        self.mutation_prob = mut_prob
        self.mut_type = mut_type
        self.crossover_prob = cross_prob
        self.cross_type = cross_type
        self.elitism = elitism
        self.population = None
        self.bin_length = len(self.data)

        self._check_parameters()

    def _check_parameters(self):
        if self.data is None or self.bin_length < 4 or \
                self.fitness_func is None or \
                self.optim not in ['min', 'max'] or \
                self.mutation_prob < 0 or self.mutation_prob > 100 or \
                self.mut_type < 1 or self.mut_type > self.bin_length or \
                self.crossover_prob < 0 or self.crossover_prob > 100 or \
                self.cross_type < 1 or self.cross_type > self.bin_length or \
                self.selection not in ["rank", "roulette", "tournament"] or \
                (self.selection == 'tournament' and self.tournament_size is None) or \
                self.elitism not in [True, False]:
            print('Wrong value of input parameter.')
            raise ValueError

    def _random_diff(self, stop, n, start=0):
        """
        Creates a list of 'n' different random numbers within the interval (start, stop) ('start' included).

        Args:
            start (int): Start value of an interval (included). Default is 0.
            stop (int): End value of an interval (excluded).
            n (int): How many different random numbers generate.

        Returns:
             list of different random values from the given interval ('start' included)
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
        This function mutates the appropriate bits from bit_num of the individual
        with the specified mutation probability.

        Args:
            individ (list): Binary encoded individual (it contains positions of bit 1 according to self.data).
            bit_num (list): List of bits' numbers to invert.

        Returns:
            mutated individual as binary representation (list)
        """
        for bit in bit_num:
            if random.uniform(0, 100) <= self.mutation_prob:
                # mutate
                if bit in individ:
                    # 1 -> 0
                    individ.remove(bit)
                else:
                    # 0 -> 1
                    individ.append(bit)

        return individ

    def _mutate(self, individ):
        """
        This function mutates (inverse bits) the given population individual.

        Args:
            individ (list): Binary encoded combination of the original data list
                (it contains positions of bit 1 according to self.data).

        Returns:
             mutated individual as binary representation (with inverted bits)
        """
        if self.bin_length == self.mut_type:
            # it is necessary to mutate all bits with the specified mutation probability
            individ = self._invert_bit(individ, list(range(self.bin_length)))
        else:
            # mutate some bits (not all)
            inverted_bits = self._random_diff(self.bin_length, self.mut_type)
            individ = self._invert_bit(individ, inverted_bits)

        return individ

    def _replace_bits(self, source, target, start, stop):
        """
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
        if start < 0 or start >= self.bin_length or \
                stop < 0 or stop < start or stop >= self.bin_length:
            print('Interval error:', '(' + str(start) + ', ' + str(stop) + ')')
            raise ValueError

        if start == stop:
            if random.uniform(0, 100) <= self.crossover_prob:
                # crossover
                if start in source:
                    # bit 'start' is 1 in source
                    if start not in target:
                        # bit 'start' is 0 in target
                        target.append(start)
                else:
                    # bit 'start' is 0 in source
                    if start in target:
                        # bit 'start' is 1 in target
                        target.remove(start)
        else:
            tmp_target = [0] * self.bin_length
            tmp_source = [0] * self.bin_length
            for index in target:
                tmp_target[index] = 1
            for index in source:
                tmp_source[index] = 1

            if random.uniform(0, 100) <= self.crossover_prob:
                # crossover
                tmp_target[start : stop+1] = tmp_source[start : stop+1]

            target = []
            for i in range(self.bin_length):
                if tmp_target[i] == 1:
                    target.append(i)

        return target

    def _cross(self, parent1, parent2):
        """
        This function crosses the two given population individuals (parents).

        Args:
            parent1 (list): binary encoded combination of the original data list (self.data) of the first parent.
            parent2 (list): binary encoded combination of the original data list (self.data) of the second parent.

        Returns:
             list: an individual (binary representation) created by crossover of two parents
        """
        new_individ = list(parent1)

        if self.cross_type == self.bin_length:
            # it is necessary to replace all bits with the specified crossover probability
            for bit in range(self.bin_length):
                new_individ = self._replace_bits(parent2, new_individ, bit, bit)
        elif self.cross_type == 1:
            # combine two parts of parents
            random_bit = random.randrange(1, self.bin_length - 1)  # we want to do useful replacements
            new_individ = self._replace_bits(parent2, new_individ, random_bit + 1, self.bin_length - 1)
        elif self.cross_type == 2:
            # replace bits within  an interval of two random generated points
            random_bit1 = random.randrange(self.bin_length)  # we want to do useful replacements
            random_bit2 = random_bit1

            while random_bit2 == random_bit1:
                random_bit2 = random.randrange(self.bin_length)

            if random_bit1 < random_bit2:
                new_individ = self._replace_bits(parent2, new_individ, random_bit1, random_bit2)
            else:
                new_individ = self._replace_bits(parent2, new_individ, random_bit2, random_bit1)
        else:
            # cross some bits exactly (not replacement within an interval)
            cross_bits = self._random_diff(self.bin_length, self.cross_type)

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
        if size > len(population) or size < 1:
            print('Wrong tournament size:', size)
            raise ValueError

        competitors = self._random_diff(len(population), size)
        # sort by fitness value in the ascending order (maximization) or descending order (minimization)
        # and get the last two elements (winner and the second best participant)
        if self.optim == 'max':
            # ascending order (maximization)
            competitors.sort(key=lambda x: population[x].fitness_val)
        else:
            # descending order (minimization)
            competitors.sort(key=lambda x: population[x].fitness_val, reverse=True)

        return competitors[-1], competitors[-2]

    def _select_parents(self, population, wheel_sum=None):
        """
        Selects parents from the given population (sorted in ascending order).

        Args:
            population (list): Current population, sorted in ascending order, from which parents will be selected.
                Population element is an IndividualGA object.
            wheel_sum (int): Sum of values on a wheel (different for "roulette" and "rank").

        Returns:
            parents (tuple of IndividualGA): selected parents
        """
        if self.selection == 'roulette' or self.selection == 'rank':
            if wheel_sum is None or wheel_sum < 2:
                print('Wrong value of wheel sum:', wheel_sum)
                raise ValueError

            parent1 = None
            parent2 = None
            random1 = random.randrange(wheel_sum)
            random2 = random.randrange(wheel_sum)

            wheel = 0
            for ind in population:
                # population is sorted in ascending order
                if self.selection == 'roulette':
                    wheel += ind.fitness_val
                else:
                    # each rank is greater by 1 than the previous one
                    wheel += wheel + 1

                if random1 < wheel:
                    parent1 = ind
                if random2 < wheel:
                    parent2 = ind

                if parent1 and parent2:
                    break

            return parent1, parent2
        elif self.selection == 'tournament':
            best1, second1 = self._conduct_tournament(population, self.tournament_size)
            best2, second2 = self._conduct_tournament(population, self.tournament_size)

            if best1.individ == best2.individ:
                return population[best1], population[second2]
            else:
                return population[best1], population[best2]
        else:
            print('Unknown selection type:', self.selection)
            raise ValueError

    def _get_bit_positions(self, number):
        """
        This function gets a decimal integer number and returns positions of bit 1 in
        its binary representation. However, these positions are transformed the following way: they
        are mapped on the data list (self.data) "as is". It means that LSB (least significant bit) is
        mapped on the last position of the data list (e.g. self.bin_length - 1), MSB is mapped on
        the first position of the data list (e.g. 0) and so on.

        Args:
            number (int): This decimal number represents binary encoded combination of the input data (self.data).

        Returns:
             list of positions with bit 1 (these positions are mapped on the input data list "as is" and thus,
             LSB is equal to index (self.bin_length - 1) of the input data list).
        """
        binary_list = []

        for i in range(self.bin_length):
            if number & (1 << i):
                binary_list.append(self.bin_length - 1 - i)

        return binary_list

    def _sort_population(self):
        if self.optim == 'max':
            # an algorithm maximizes a fitness value
            # ascending order
            self.population.sort(key=lambda x: x.fitness_val)
        else:
            # an algorithm minimizes a fitness value
            # descending order
            self.population.sort(key=lambda x: x.fitness_val, reverse=True)

    def init_population(self, new_population):
        """
        Initializes population with the value of 'new_population'.

        Args:
            new_population (list): New initial population of IndividualGA objects.
        """
        if not new_population:
            print('New population is empty')
            raise ValueError

        self.population = []
        for individ in new_population:
            fit_val = self.fitness_func(individ, self.data)
            self.population.append(IndividualGA(individ, fit_val))

        self._sort_population()

    def init_random_population(self, size=None):
        """
        Initializes a new random population of the given size.

        Args:
            size (int): Size of a new random population. If None, the size is set to (2**self.bin_length) / 10, because
                self.bin_length is a number of bits. Thus, a new population of size 10% of all possible solutions
                (or of size 4 in case of self.bin_length < 5) will be created.
        """
        max_num = 2 ** self.bin_length

        if size is None:
            if max_num < 20:
                size = 4
            else:
                # 10% of all possible solutions
                size = max_num // 10
        elif 2 > size > self.bin_length:
            print('Wrong size of population:', size)
            raise ValueError

        # generate population
        number_list = self._random_diff(max_num, size, start=1)

        self.population = []
        for num in number_list:
            individ = self._get_bit_positions(num)
            fit_val = self.fitness_func(individ, self.data)

            self.population.append(IndividualGA(individ, fit_val))

        self._sort_population()

    def run(self, max_generation):
        """
        Starts GA. The algorithm does 'max_generation' generations and then stops.
        Old population is completely replaced with new one.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            list of lists of fitness values by generation
        """
        fitness_progress = [[ind.fitness_val for ind in self.population]]

        for generation_num in range(max_generation):
            fitness_sum = sum(ind.fitness_val for ind in self.population)
            population_size = len(self.population)
            next_population = []

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
                # compute fitness value of a child
                fit_val = self.fitness_func(new_individ, self.data)

                next_population.append(IndividualGA(new_individ, fit_val))

            if self.elitism:
                # copy the best individual to a new generation
                next_population.append(self.population[-1])

            self.population = next_population
            self._sort_population()

            fitness_progress.append([ind.fitness_val for ind in self.population])

        return fitness_progress

