import random


class IndividualGA:
    """
    The class represents an individual of population in GA.
    """
    def __init__(self, individ, fitness_val):
        """
        A constructor.

        Args:
            individ (list): A chromosome represented a solution. The solution is binary encoded in chromosome.
            fitness_val (int): Fitness value of the given chromosome.
        """
        self.individ = list(individ)
        self.fitness_val = fitness_val


class GeneticAlgorithms:
    def __init__(self, data=None, fitness_func=None, selection="rank", mut_prob=0.5, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=False, tournament_size=None):
        """
        Args:
            data (list): A list with elements whose combination will be binary encoded and
                evaluated by a fitness function. Minimum amount of elements is 4.
            fitness_func (function): This function must compute fitness value of an input combination of the given data.
                Function parameters must be: list of used indices of the given data (from 0), list of data itself.
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
        self.selection = selection
        self.tournament_size = tournament_size
        self.mutation_prob = mut_prob
        self.mut_type = mut_type
        self.crossover_prob = cross_prob
        self.cross_type = cross_type
        self.elitism = elitism
        self.population = None

        self._check_parameters()

    def _check_parameters(self):
        if self.data is None or len(self.data) < 4 or \
                self.fitness_func is None or \
                self.mutation_prob < 0 or self.mutation_prob > 100 or \
                self.mut_type < 1 or self.mut_type > len(self.data) or \
                self.crossover_prob < 0 or self.crossover_prob > 100 or \
                self.cross_type < 1 or self.cross_type > len(self.data) or \
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
        random_number = random.randrange(start, stop)
        used_values = [random_number]

        for i in range(1, n):
            while random_number in used_values:
                random_number = random.randrange(start, stop)

            used_values.append(random_number)

        return used_values

    def _invert_bit(self, bit_value):
        """
        This function inverts the given boolean value with the specified mutation probability.

        Args:
            bit_value (True, False): The value to invert.
        Returns:
            inverted value: True for False, False for True
        """
        if random.uniform(0, 100) <= self.mutation_prob:
            return not bit_value
        else:
            # don't mutate
            return bit_value

    def _mutate(self, individ):
        """
        This function mutates (inverse bits) the given population individual.

        Args:
            individ (list): binary encoded combination of input data.
        Returns:
             mutated individual (with inverted bits)
        """
        if len(individ) == self.mut_type:
            # it is necessary to mutate all bits with the specified mutation probability
            for i in range(len(individ)):
                individ[i] = self._invert_bit(individ[i])
        else:
            # mutate some bits (not all)
            inverted_bits = self._random_diff(len(individ), self.mut_type)

            for bit in inverted_bits:
                individ[bit] = self._invert_bit(individ[bit])

        return individ

    def _replace_bits(self, source, target, start, stop):
        """
        Replace target bits with source bits in interval (start, stop) (both included)
        with the specified crossover probability.

        Args:
            source (list): Values in source are used as replacement for target.
            target (list): Values in target are replaced with values in source.
            start (int): Start point of interval (included).
            stop (int): End point of interval (included).
        Returns:
             target with replaced bits with source one in the interval (start, stop) (both included)
        """
        if start < 0 or start >= len(source) or \
                stop < 0 or stop < start or stop >= len(source):
            print('Interval error:', '(' + str(start) + ', ' + str(stop) + ')')
            raise ValueError

        if random.uniform(0, 100) <= self.crossover_prob:
            for i in range(start, stop + 1):
                target[i] = source[i]

        return target

    def _cross(self, parent1, parent2):
        """
        This function crosses the two given population individuals (parents).

        Args:
            parent1 (list): binary encoded combination of input data of the first parent.
            parent2 (list): binary encoded combination of input data of the second parent.
        Returns:
             list: individual created by crossover of two parents
        """
        new_individ = list(parent1)

        if self.cross_type == len(parent1):
            # it is necessary to replace all bits with the specified crossover probability
            for bit in range(len(parent1)):
                new_individ = self._replace_bits(parent2, new_individ, bit, bit)
        elif self.cross_type == 1:
            # combine two parts of parents
            random_bit = random.randrange(1, len(parent2) - 1)  # we want to do useful replacements
            new_individ = self._replace_bits(parent2, new_individ, random_bit + 1, len(parent2) - 1)
        elif self.cross_type == 2:
            # replace bits within  an interval of two random generated points
            random_bit1 = random.randrange(len(parent1))  # we want to do useful replacements
            while random_bit2 == random_bit1:
                random_bit2 = random.randrange(len(parent1))

            if random_bit1 < random_bit2:
                new_individ = self._replace_bits(parent2, new_individ, random_bit1, random_bit2)
            else:
                new_individ = self._replace_bits(parent2, new_individ, random_bit2, random_bit1)
        else:
            # cross some bits exactly (not replacement within an interval)
            cross_bits = self._random_diff(len(parent2), self.cross_type)

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
            winners (tuple): winner of current tournament and the second best participant
        """
        if size > len(population) or size < 1:
            print('Tournament size is greater than the whole population.')
            raise ValueError

        competitors = self._random_diff(len(population), size)
        # sort by fitness value in the ascending order
        # and get the last two elements (winner and the second best participant)
        competitors.sort(key=lambda x: population[x].fitness_val)

        return competitors[-1], competitors[-2]

    def _select_parents(self, population, wheel_sum=None):
        """
        Selects parents from the given population.

        Args:
            population (list): Current population from which parents will be selected.
                Population element is an IndividualGA object.
            wheel_sum (int): Sum of values on a wheel (different for "roulette" and "rank").
        Returns:
            parents (tuple): selected parents
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
                wheel += ind.fitness_val

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
                return best1, second2
            else:
                return best1, best2
        else:
            print('Unknown selection type:', self.selection)
            raise ValueError

    def _get_bit_positions(self, number):
        """
        This function gets a decimal integer number and returns positions of bit 1 in
        its binary representation. However, these positions are transformed the following way: they
        are mapped on the data list (self.data) "as is". It means that LSB (least significant bit) is
        mapped on the last position of the data list (e.g. len(self.data) - 1), MSB is mapped on
        the first position of the data list (e.g. 0) and so on.

        Args:
            number (int): This decimal number represents binary encoded combination of the input data (self.data).
        Returns:
             list of positions with bit 1 (these positions are mapped on the input data list "as is" and thus,
             LSB is equal to index (len(self.data) - 1) of the input data list).
        """
        binary_list = []
        number_of_bits = len(self.data)

        for i in range(number_of_bits):
            if number & (1 << i):
                binary_list.append(number_of_bits - 1 - i)

        return binary_list

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

    def init_random_population(self, size=None):
        """
        Initializes a new random population of the given size.

        Args:
            size (int): Size of a new random population. If None, the size is set to (2**len(self.data)) / 10, because
                len(self.data) is a number of bits. Thus, a new population of size 10% of all possible solutions
                (or of size 4 in case of len(self.data) < 5) will be created.
        """
        max_num = 2 ** len(self.data)

        if size is None:
            if max_num < 20:
                size = 4
            else:
                # 10% of all possible solutions
                size = max_num // 10
        elif 2 > size > len(self.data):
            print('Wrong size of population:', size)
            raise ValueError

        # generate population
        number_list = self._random_diff(max_num, size, start=1)

        self.population = []
        for num in number_list:
            individ = self._get_bit_positions(num)
            fit_val = self.fitness_func(individ, self.data)

            self.population.append(IndividualGA(individ, fit_val))

    def run(self):
        # fitness_sum = sum(ind.fitness_val for ind in population)
        # TODO
        pass



