import random


class IndividualGA:
    """
    The class represents an element of population in GA.
    """
    def __init__(self, individ, fitness_val):
        """

        Args:
            fitness_val (int): Fitness value of the given chromosome.
        """
        self.individ = individ
        self.fitness_val = fitness_val


class GeneticAlgorithms:
    def __init__(self, data, selection="rank", mut_prob=0.5, mut_type=1, cross_prob=0.95, cross_type=1, elitism=False,
                 tournament_size=None):
        """
        Args:
            data (list): A list with elements of the original population. This list will be binary encoded
                (with True, False) later in order to indicate currently evaluated combination of elements.
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
        self.selection = selection
        self.tournament_size = tournament_size
        self.mutation_prob = mut_prob
        self.mut_type = mut_type
        self.crossover_prob = cross_prob
        self.cross_type = cross_type
        self.elitism = elitism

        self._check_parameters()

    def _check_parameters(self):
        if len(self.data) < 2 or \
                self.mutation_prob < 0 or self.mutation_prob > 100 or \
                self.mut_type < 1 or self.mut_type > len(self.data) or \
                self.crossover_prob < 0 or self.crossover_prob > 100 or \
                self.cross_type < 1 or self.cross_type > len(self.data) or \
                self.selection not in ["rank", "roulette", "tournament"] or \
                (self.selection == 'tournament' and self.tournament_size is None) or \
                self.elitism not in [True, False]:
            print('Wrong value of input parameter.')
            raise ValueError

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
            random_bit = random.randrange(len(individ))
            individ[random_bit] = self._invert_bit(individ[random_bit])

            inverted_bits = [random_bit]
            for i in range(1, self.mut_type):
                while random_bit in inverted_bits:
                    random_bit = random.randrange(len(individ))

                inverted_bits.append(random_bit)
                individ[random_bit] = self._invert_bit(individ[random_bit])

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
            random_bit = random.randrange(len(parent2))
            new_individ = self._replace_bits(parent2, new_individ, random_bit, random_bit)

            cross_bits = [random_bit]
            for i in range(1, self.cross_type):
                while random_bit in cross_bits:
                    random_bit = random.randrange(len(parent2))

                cross_bits.append(random_bit)
                new_individ = self._replace_bits(parent2, new_individ, random_bit, random_bit)

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

        participant = random.randrange(len(population))
        competitors = [participant]

        for i in range(1, size):
            while participant in competitors:
                participant = random.randrange(len(population))

            competitors.append(participant)

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
            if wheel_sum is None:
                print('Wheel sum is unknown: cannot continue')
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

    def _init_population(self):
        pass

    def run(self):
        # fitness_sum = sum(ind.fitness_val for ind in population)
        pass

