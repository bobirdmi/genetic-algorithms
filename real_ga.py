from bitstring import BitArray
import random
import numpy

from genetic_algorithms import GeneticAlgorithms, IndividualGA


class RealGeneticAlgorithms(GeneticAlgorithms):
    def __init__(self, fitness_func=None, optim='max', type='standard', selection="rank", mut_prob=0.05, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=True, tournament_size=None):
        """
        Args:
            fitness_func (function): This function must compute fitness value of a single individual.
                Function parameter must be: a single individual.
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
        super().__init__(fitness_func, optim, type, selection,
                         mut_prob, mut_type, cross_prob, cross_type,
                         elitism, tournament_size)
        self._bin_length = 64

        self._check_parameters()

        self._mut_bit_offset = self._get_mut_bit_offset()

    def _get_mut_bit_offset(self):
        """
        Returns bit number (from left to the right) in 32- or 64-bit big-endian floating point 
        binary representation (IEEE 754) from which a mantissa begins. It is necessary because this real GA implementation 
        mutate only mantissa bits (mutation of exponent changes a float number the undesired fast and unexpected way).
        """
        # IEEE 754
        # big-endian floating point binary representation
        # | sign | exponent | mantissa |
        # | 1 | 8 | 23 | in 32-bit floating point
        # | 1 | 11 | 52 | in 64-bit floating point
        if self._bin_length == 32:
            return 1 + 8
        elif self._bin_length == 64:
            return 1 + 11
        else:
            print('Wrong floating point binary length: may be only 32 or 64.')
            raise ValueError

    def _check_parameters(self):
        if self._bin_length not in [32, 64] or \
                self.mut_type > self._bin_length or \
                self.cross_type > self._bin_length:
            print('Wrong value of input parameter.')
            raise ValueError

    def _is_individ_list(self, individ):
        """
        This function returns True iff individ is a list (even list of just 1 element),
        otherwise False.

        Args:
            individ (float, list): An individual of GA population. May be float or a list of floats.

        Returns:
            True iff the given individual is a list (even a list of just 1 element), otherwise False.
        """
        try:
            list(individ)
            return True  # it is a list
        except TypeError:
            return False  # it is a single number

    def _get_individ_return_value(self, individ):
        """
        This function returns a vector (individual as list of floats) or a single float
        depending on number of elements in the given individual.

        Args:
            individ (list): This list contains a single float or represents a vector of floats.

        Returns:
            individ[0] iff there is only 1 element in the list, otherwise individ
        """
        if len(individ) > 1:
            return individ
        else:
            return individ[0]

    def _invert_bit(self, individ, bit_num):
        """
        This function mutates the appropriate bits from bit_num of the individual
        with the specified mutation probability. The function mutates bit_num's bits of all floats
        in a list represented individual in case of multiple dimensions.

        Args:
            individ (float, list): A single float or a list of floats in case of multiple dimensions.
            bit_num (list): List of bits' numbers to invert.

        Returns:
            mutated individual (float, list)
        """
        mutated_individ = []

        is_vector = self._is_individ_list(individ)
        if is_vector:
            origin_individ = individ
        else:
            # it is a single float, not a list
            origin_individ = [individ]

        for ind in origin_individ:
            bstr = BitArray(floatbe=ind, length=self._bin_length)

            for bit in bit_num:
                if random.uniform(0, 1) <= self.mutation_prob:
                    # mutate
                    bstr[bit] = not bstr[bit]

            mutated_individ.append(bstr.floatbe)

        return self._get_individ_return_value(mutated_individ)

    def _replace_bits(self, source, target, start, stop):
        """
        Replace target bits with source bits in interval (start, stop) (both included)
        with the specified crossover probability.

        Args:
            source (float, list): Values in source are used as replacement for target. May be float or list of floats
                in case of multiple dimensions.
            target (float, list): Values in target are replaced with values in source. May be float or list of floats
                in case of multiple dimensions.
            start (int): Start point of an interval (included).
            stop (int): End point of an interval (included).

        Returns:
             target (float, list) with replaced bits with source one in the interval (start, stop) (both included)
        """
        if start < 0 or start >= self._bin_length or \
                stop < 0 or stop < start or stop >= self._bin_length:
            print('Interval error:', '(' + str(start) + ', ' + str(stop) + ')')
            raise ValueError

        is_vector = self._is_individ_list(source)
        if is_vector:
            origin_source = source
            origin_target = target
        else:
            # it is a single float, not a list
            origin_source = [source]
            origin_target = [target]

        child = []
        for source_ind, target_ind in zip(origin_source, origin_target):
            bstr_source = BitArray(floatbe=source_ind, length=self._bin_length)
            bstr_target = BitArray(floatbe=target_ind, length=self._bin_length)

            if random.uniform(0, 1) <= self.crossover_prob:
                # crossover
                if start == stop:
                    bstr_target[start] = bstr_source[start]
                else:
                    bstr_target[start: stop + 1] = bstr_source[start: stop + 1]

            child.append(bstr_target.floatbe)

        return self._get_individ_return_value(child)

    def _compute_fitness(self, individ):
        """
        This function computes fitness value of the given individual.

        Args:
            individ (float, list): An individual of genetic algorithm. May be a single float
                or a list of floats in case of multiple dimensions. Defined fitness function (self.fitness_func)
                must deal with this individual.

        Returns:
            fitness value of the given individual
        """
        return self.fitness_func(individ)

    def init_random_population(self, size, dim, interval):
        """
        Initializes a new random population of the given size with individual's values
        within the given interval (start point included, end point excluded)
        with the given amount of dimensions.

        Args:
            size (int): Size of a new random population.
            dim (int): Amount of space dimensions.
            interval (tuple): The generated numbers of each dimension will be 
                within this interval (start point included, end point excluded).
                Both end points must be integer values.
        """
        if size < 2 or dim < 1 or interval[0] >= interval[1]:
            print('Wrong value of input parameter.')
            raise ValueError

        if dim > 1:
            self._is_vector = True

        # generate population
        individs = numpy.random.uniform(int(interval[0]), int(interval[1]), (size, dim))

        if self.type == 'standard':
            self._population = []
            for ind in individs:
                if dim == 1:
                    individ = ind[0]
                else:
                    individ = ind

                fit_val = self._compute_fitness(individ)

                self._population.append(IndividualGA(individ, fit_val))

            self._sort_population()
            self._update_solution(self._population[-1].individ, self._population[-1].fitness_val)
        elif self.type == 'diffusion':
            if dim == 1:
                population = [ind[0] for ind in individs]
            else:
                population = individs

            self._init_diffusion_model(population)
        elif self.type == 'migration':
            # TODO migration model
            pass