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


from bitstring import BitArray
import random
import numpy

from .standard_ga import StandardGA, IndividualGA


class RealGA(StandardGA):
    """
    This class realizes GA over the real values. In other words, it tries to find global minimum or
    global maximum (depends on the settings) of a given fitness function.

    Attributes:
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

    You may initialize instance of this class the following way

    .. testcode::

       from geneticalgs import RealGA
       import math

       # define some function whose global minimum or maximum we are searching for
       # this function takes as input one-dimensional number
       def fitness_function(x):
           # the same function is used in examples
           return abs(x*(math.sin(x/11)/5 + math.sin(x/110)))

       # initialize standard real GA with fitness maximization by default
       gen_alg = RealGA(fitness_function)
       # initialize random one-dimensional population of size 20 within interval (0, 1000)
       gen_alg.init_random_population(20, 1, (0, 1000))

    Then you may start computation by *gen_alg.run(number_of_generations)* and obtain
    the currently best found solution by *gen_alg.best_solution*.
    """
    def __init__(self, fitness_func=None, optim='max', selection="rank", mut_prob=0.05, mut_type=1,
                 cross_prob=0.95, cross_type=1, elitism=True, tournament_size=None):
        """
        Args:
            fitness_func (function): This function must compute fitness value of a single real value chromosome.
                The returned value of the this fitness function must be a single number.
            optim (str): What this genetic algorithm must do with fitness value: maximize or minimize.
                May be 'min' or 'max'. Default is "max".
            selection (str): Parent selection type. May be "rank" (Rank Wheel Selection),
                "roulette" (Roulette Wheel Selection) or "tournament". Default is "rank".
            tournament_size (int): Defines the size of tournament in case of 'selection' == 'tournament'.
                Default is None.
            mut_prob (float): Probability of mutation. Recommended values are 0.5-1%. Default is 0.5% (0.05).
            mut_type (int): This parameter defines how many chromosome bits will be mutated.
                May be 1 (single-point), 2 (two-point), 3 or more (multiple point). Default is 1.
            cross_prob (float): Probability of crossover. Recommended values are 80-95%. Default is 95% (0.95).
            cross_type (int): This parameter defines crossover type. The following types are allowed:
                single point (1), two point (2) and multiple point (2 < cross_type).
                The extreme case of multiple point crossover is uniform one (cross_type == all_bits).
                The specified number of bits (cross_type) are crossed in case of multiple point crossover.
                Default is 1.
            elitism (True, False): Elitism on/off. Default is True.
        """
        super().__init__(fitness_func, optim, selection,
                         mut_prob, mut_type, cross_prob, cross_type,
                         elitism, tournament_size)
        self._bin_length = 64  # may be only 32 or 64

        self._check_parameters()

        self._mut_bit_offset = self._get_mut_bit_offset()
        self.interval = None

    def _get_mut_bit_offset(self):
        """
        Returns bit number (from left (index 0) to the right) in 32- or 64-bit big-endian floating point
        binary representation (IEEE 754) from which a mantissa begins. It is necessary because this real GA implementation 
        mutates only mantissa bits (mutation of exponent changes a float number the undesired fast and unexpected way).
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
            raise ValueError('Wrong floating point binary length: may be only 32 or 64.')

    def _check_parameters(self):
        if self._bin_length not in [32, 64] or \
                self.mut_type > self._bin_length or \
                self.cross_type > self._bin_length:
            raise ValueError('Wrong value of input parameter.')

    def _is_chromosome_list(self, chromosome):
        """
        This method returns True iff chromosome is a list (even list of just 1 element),
        otherwise False.

        Args:
            chromosome (float, list): A chromosome of GA population. May be float or a list of floats
                in case of multiple dimensions.

        Returns:
            True iff the given chromosome is a list (even a list of just 1 element), otherwise False.
        """
        try:
            list(chromosome)
            return True  # it is a list
        except TypeError:
            return False  # it is a single number

    def _get_chromosome_return_value(self, chromosome):
        """
        This method returns a vector (chromosome as a list of floats) or a single float
        depending on number of elements in the given chromosome.

        Args:
            chromosome (list): This list contains a single float or represents a vector of floats
                in case of multiple dimensions.

        Returns:
            *chromosome[0]* iff there is only 1 element in the list, otherwise *chromosome*
        """
        try:
            length = len(chromosome)

            if length < 1:
                raise ValueError('The given chromosome is empty!')
            elif length > 1:
                return chromosome
            else:
                return chromosome[0]
        except TypeError:
            raise ValueError('The given chromosome is not a list!')

    def _adjust_to_interval(self, var):
        """
        This method replaces NaN, inf, -inf in *var* by numpy.nan_to_num() and then
        returns *var* if it is within the specified interval. Otherwise returns lower bound of the interval
        if (*var* < lower bound) or upper bound of the interval if (*var* > upper bound).

        Args:
            var (list, float): A float or a list of floats to adjust to the specified interval.

        Returns:
            adjusted input parameter
        """
        var = numpy.nan_to_num(var)

        try:
            dim = len(var)

            for num, d in zip(var, range(dim)):
                var[d] = max(min(self.interval[1], num), self.interval[0])
        except TypeError:
            var = max(min(self.interval[1], var), self.interval[0])

        return var

    def _invert_bit(self, chromosome, bit_num):
        """
        This method mutates the appropriate bits of the chromosome from *bit_num*
        with the specified mutation probability. The method mutates bit_num's bits of all floats
        in a list represented chromosome in case of multiple dimensions.

        Args:
            chromosome (float, list): A single float or a list of floats in case of multiple dimensions.
            bit_num (list): List of bits' numbers to invert.

        Returns:
            mutated chromosome (float, list)
        """
        mutated_chromosome = []

        is_vector = self._is_chromosome_list(chromosome)
        if is_vector:
            origin_chromosome = chromosome
        else:
            # it is a single float, not a list
            origin_chromosome = [chromosome]

        for chrom in origin_chromosome:
            bstr = BitArray(floatbe=chrom, length=self._bin_length)

            for bit in bit_num:
                if random.uniform(0, 1) <= self.mutation_prob:
                    # mutate
                    bstr[bit] = not bstr[bit]

            mutated_chromosome.append(bstr.floatbe)

        return self._adjust_to_interval(self._get_chromosome_return_value(mutated_chromosome))

    def _replace_bits(self, source, target, start, stop):
        """
        Replaces target bits with source bits in interval (start, stop) (both included)
        with the specified crossover probability. This interval represents
        positions of bits to replace (minimum start point is 0 and maximum end point is *self._bin_length - 1*).

        Args:
            source (float, list): Values in source are used as replacement for target. May be a float or a list of floats
                in case of multiple dimensions.
            target (float, list): Values in target are replaced with values in source. May be a float or a list of floats
                in case of multiple dimensions.
            start (int): Start point of interval (included).
            stop (int): End point of interval (included).

        Returns:
             target (float, list): Target with replaced bits with source one in the interval (start, stop) (both included).
        """
        if start < 0 or start >= self._bin_length or \
                stop < 0 or stop < start or stop >= self._bin_length:
            print('Interval error:', '(' + str(start) + ', ' + str(stop) + ')')
            raise ValueError('Replacement interval error')

        is_vector = self._is_chromosome_list(source)
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

        return self._adjust_to_interval(self._get_chromosome_return_value(child))

    def _compute_fitness(self, chromosome):
        """
        This method computes fitness value of the given chromosome.

        Args:
            chromosome (float, list): A chromosome of genetic algorithm. May be a single float
                or a list of floats in case of multiple dimensions. Defined fitness function (*self.fitness_func*)
                must deal with this chromosome representation.

        Returns:
            fitness value of the given chromosome
        """
        return self.fitness_func(chromosome)

    def _check_init_random_population(self, size, dim, interval):
        """
        This method verifies the input parameters of a random initialization.

        Args:
            size (int): Size of a new population.
            dim (int): Amount of space dimensions.
            interval (tuple): The generated numbers of each dimension will be
                within this interval (start point included, end point excluded).
                Both end points must be *different* integer values.
        """
        if size is None or dim is None or interval is None or \
                        size < 2 or dim < 1 or interval[0] >= interval[1]:
            raise ValueError('Wrong value of input parameter.')

    def _generate_random_population(self, size, dim, interval):
        """
        This method generates a new random population by the given input parameters.

        Args:
            size (int): Size of a new population.
            dim (int): Amount of space dimensions.
            interval (tuple): The generated numbers of each dimension will be
                within this interval (start point included, end point excluded).

        Returns:
            array (numpy.array): Array rows represent chromosomes. Number of columns is specified
            with *dim* parameter.
        """
        self.interval = interval
        return numpy.random.uniform(interval[0], interval[1], (int(size), int(dim)))

    def init_random_population(self, size, dim, interval):
        """
        Initializes a new random population of the given size with chromosomes' values
        within the given interval (start point included, end point excluded)
        with the specified amount of dimensions.

        Args:
            size (int): Size of a new random population. Must be at least 2.
            dim (int): Amount of space dimensions.
            interval (tuple): The generated numbers of each dimension will be 
                within this interval (start point included, end point excluded).
        """
        self._check_init_random_population(size, dim, interval)

        # generate population
        chromosomes = self._generate_random_population(size, dim, interval)

        self.population = []
        for chrom in chromosomes:
            if dim == 1:
                chromosome = chrom[0]
            else:
                chromosome = chrom

            fit_val = self._compute_fitness(chromosome)

            self.population.append(IndividualGA(chromosome, fit_val))

        self._sort_population()
        self._update_solution(self.population[-1].chromosome, self.population[-1].fitness_val)
