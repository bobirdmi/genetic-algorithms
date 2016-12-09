import random


class GeneticAlgorithms:
    def __init__(self, data, selection="rank", mut_prob=0.5, mut_type=1, cross_prob=0.95, cross_type=1, elitism=False):
        """
        Args:
            data (list): A list with elements of the original population. This list will be binary encoded
                (with True, False) later in order to indicate currently evaluated combination of elements.
            selection (str): Parent selection type. May be "rank" (Rank selection), "roulette" (Roulette Wheel Selection)
            mut_prob (float): Probability of mutation. Recommended values are 0.5-1%.
            mut_type (int): This parameter defines how many bits of individual are inverted (mutated).
            cross_prob (float): Probability of crossover. Recommended values are 80-95%.
            cross_type (int): The following types of crossover are allowed: single point (1),
                two point (2) and uniform (3).
            elitism (True, False): Elitism on/off.
        """
        self.data = data
        self.selection = selection
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
                self.selection not in ["rank", "roulette"] or \
                self.elitism not in [True, False]:
            raise ValueError

    def _invert_bit(self, bit_value):
        """
        This function inverts the given boolean value.

        Args:
            bit_value (True, False): The value to invert.
        Returns:
            inverted value: True for False, False for True
        """
        return not bit_value

    def _mutate(self, individ):
        if random.uniform(0, 100) > self.mutation_prob:
            # don't mutate
            return individ

        random_bit = random.randrange(len(individ))
        individ[random_bit] = self._invert_bit(individ[random_bit])

        inverted_bits = [random_bit]
        for i in range(1, self.mut_type):
            while random_bit in inverted_bits:
                random_bit = random.randrange(len(individ))

            inverted_bits.append(random_bit)
            individ[random_bit] = self._invert_bit(individ[random_bit])

        return individ

    def _cross(self, individ1, individ2):
        pass

    def _select_parents(self, population):
        pass

