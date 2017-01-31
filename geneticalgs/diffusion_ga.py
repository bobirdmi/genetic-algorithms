"""
   Copyright 2017 Dmitriy Bobir <bobirdima@gmail.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import numpy
import math

from .standard_ga import IndividualGA


TYPE_BINARY = 0
TYPE_REAL = 1


class DiffusionGA:
    """
    This class implements diffusion model of genetic algorithms. The current implementation supports
    four neighbours (up, down, left, right) of a currently processed cell. Supports the standard selection types
    (e.g. "rank", "roulette", "tournament"). It's evident that the maximum tournament size is 4 in this case.
    """
    def __init__(self, instance):
        """
        A constructor.

        Args:
            instance (BinaryGA, RealGA): An instance of Binary Genetic Algorithm or of Real GA.
                Type of this instance (binary or real GA) determines behaviour of a diffusion model.
        """
        self._ga = instance

        if hasattr(self._ga, '_data'):
            self.type = TYPE_BINARY
        else:
            self.type = TYPE_REAL

        self._fitness_arr = None
        self._chrom_arr = None

    @property
    def population(self):
        """
        Returns the following tuple: (array of chromosomes, array of their fitness values).

        Returns:
           array of chromosomes, array fitness values (tuple): Array of chromosomes and another array with
                their fitness values.
        """
        return self._chrom_arr, self._fitness_arr

    @property
    def best_solution(self):
        """
        Returns tuple in the following form: (best chromosome, its fitness value).

        Returns:
            tuple with the currently best found chromosome and its fitness value.
        """
        return self._ga.best_solution

    def _get_neighbour(self, row, column):
        """
        The function returns a chromosome selected from the four neighbours (up, down, left, right)
        of the currently processed cell (specified with the given row and column)
        according to the selection type ("rank", "roulette" or "tournament").

        Args:
            row (int): Row of a current cell.
            column (int): Column of a current cell.

        Returns:
            chromosome (binary encoded, float, list of floats): A chromosome selected from neighbours
                according to the specified selection type ("rank", "roulette", "tournament").
        """
        shape = self._chrom_arr.shape
        up, down, left, right = (0, 1, 2, 3)
        DIRS = {
            up: ((row - 1) % shape[0], column),
            down: ((row + 1) % shape[0], column),
            left: (row, (column - 1) % shape[1]),
            right: (row, (column + 1) % shape[1])
        }

        arr_size = len(DIRS)
        fit_arr = numpy.empty(arr_size)
        population = []

        for d, i in zip(list(DIRS.keys()), range(arr_size)):
            fit_arr[i] = self._fitness_arr[DIRS[d]]
            population.append(IndividualGA(self._chrom_arr[DIRS[d]], fit_arr[i]))

        wheel_sum = 0
        if self._ga.selection == 'rank':
            wheel_sum = self._ga._compute_rank_wheel_sum(arr_size)
        elif self._ga.selection == 'roulette':
            wheel_sum = sum(fit_arr)

        # we need only one parent not two
        return self._ga._select_parents(population, wheel_sum)[0].chromosome

    def _compute_diffusion_generation(self, chrom_arr):
        """
        This function computes a new generation of a diffusion model of GA.

        Args:
            chrom_arr (numpy.array): Diffusion array of chromosomes (binary encoded, float or a list of floats)
                of the current generation.

        Returns:
            new_chrom_array, new_fitness_arr (numpy.array, numpy.array): New diffusion arrays of chromosomes
                and their fitness values of the next generation.
        """
        shape = chrom_arr.shape
        new_chrom_arr = numpy.empty(shape, dtype=object)
        new_fitness_arr = numpy.empty(shape)

        for row in range(shape[0]):
            for column in range(shape[1]):
                parent1 = chrom_arr[row, column]
                parent2 = self._get_neighbour(row, column)

                # cross parents and mutate a child
                new_chromosome = self._ga._mutate(self._ga._cross(parent1, parent2))

                # compute fitness value of the child
                fit_val = self._ga._compute_fitness(new_chromosome)

                new_chrom_arr[row, column] = new_chromosome
                new_fitness_arr[row, column] = fit_val

        coords_best, coords_worst = self._find_critical_values(new_fitness_arr)

        if self._ga.elitism:
            # replace the worst solution in the new generation
            # with the best one from the previous generation
            new_chrom_arr[coords_worst] = self._ga.best_chromosome
            new_fitness_arr[coords_worst] = self._ga.best_fitness

        # update the best solution taking into account a new generation
        self._ga._update_solution(new_chrom_arr[coords_best], new_fitness_arr[coords_best])

        return new_chrom_arr, new_fitness_arr

    def _find_critical_values(self, fitness_arr):
        """
        Finds 1D or 2D array coordinates of the best and the worst fitness values in the given array.
        Returns coordinates of the first occurrence of these critical values.

        Args:
            fitness_arr (numpy.array): Array of fitness values.

        Returns:
            coords_best, coords_worst (tuple): Coordinates of the best and the worst
                fitness values as (index_best, index_worst) in 1D or ((row, column), (row, column)) in 2D.
        """
        # get indices of the best and the worst solutions in new generation
        # actually indices of ALL solutions with the best and the worst fitness values
        indices_max = numpy.where(fitness_arr == fitness_arr.max())
        indices_min = numpy.where(fitness_arr == fitness_arr.min())
        arr_dim = len(fitness_arr.shape)

        if arr_dim > 2:
            raise ValueError('Only 1D or 2D arrays are supported.')

        if self._ga.optim == 'min':
            # fitness minimization
            if arr_dim == 1:
                coords_worst = indices_max[0][0]
                coords_best = indices_min[0][0]
            else:
                coords_worst = (indices_max[0][0], indices_max[1][0])
                coords_best = (indices_min[0][0], indices_min[1][0])
        else:
            # fitness maximization
            if arr_dim == 1:
                coords_worst = indices_min[0][0]
                coords_best = indices_max[0][0]
            else:
                coords_worst = (indices_min[0][0], indices_min[1][0])
                coords_best = (indices_max[0][0], indices_max[1][0])

        return coords_best, coords_worst

    def _construct_diffusion_model(self, population):
        """
        Constructs two arrays: first for chromosomes of GA, second for their fitness values.
        The current implementation supports construction of only 2D square arrays. Thus, an array side is
        a square root of the given population length. If the calculated square root is a fractional number,
        it will be truncated that means the last chromosomes in population may not be
        presented in the constructed arrays.

        Args:
            population (list): List of GA chromosomes. Same as in *self.init_population(new_population)*.
        """
        size = int(math.sqrt(len(population)))

        self._chrom_arr = numpy.empty((size, size), dtype=object)
        self._fitness_arr = numpy.empty((size, size))

        index = 0
        for row in range(size):
            for column in range(size):
                self._chrom_arr[row, column] = population[index]
                self._fitness_arr[row, column] = self._ga._compute_fitness(population[index])

                index += 1

    def _init_diffusion_model(self, population):
        """
        This function constructs diffusion model from the given population
        and then updates the currently best found solution.

        Args:
            population (list): List of GA chromosomes.
        """
        self._construct_diffusion_model(population)

        coords_best, _ = self._find_critical_values(self._fitness_arr)
        self._ga._update_solution(self._chrom_arr[coords_best], self._fitness_arr[coords_best])

    def init_population(self, new_population):
        """
        Initializes population with the given chromosomes (binary encoded, float or a list of floats)
        in *new_population*. The fitness values of these chromosomes will be computed by a specified fitness function.

        It is recommended to have new_population size equal to some squared number (9, 16, 100, 625 etc.)
        in case of diffusion model of GA. Otherwise some last chromosomes in the given population will be lost
        as the current implementation supports only square arrays of diffusion model.

        Args:
            new_population (list): A new population of chromosomes of size at least 4.
                A single chromosome in case of binary GA is represented as a list of bits' positions
                with value 1 in the following way: LSB (least significant bit) has position (*len(self.data)* - 1)
                and MSB (most significant bit) has position 0. If it is a GA on real values,
                an individual is represented as a float or a list of floats in case of multiple dimensions.
        """
        if not new_population or len(new_population) < 4:
            raise ValueError('New population is too small.')

        self._init_diffusion_model(new_population)

    def init_random_population(self, size, dim=None, interval=None):
        """
        Initializes a new random population with the given parameters.

        Args:
            size (int): A size of new generated population. Must be at least 2 in case of RealGA and
                at least 4 in case of BinaryGA.
            dim (int, None): Amount of space dimensions in case of RealGA.
            interval (tuple, None): The generated numbers of each dimension will be
                within this interval (start point included, end point excluded).
                Must be specified in case of RealGA.
        """
        if self.type == TYPE_BINARY:
            # there is a binary GA
            max_num = self._ga._check_init_random_population(size)

            number_list = self._ga._generate_random_population(max_num, size)
            population = [self._ga._get_bit_positions(num) for num in number_list]
        else:
            # there is a GA on real values
            self._ga._check_init_random_population(size, dim, interval)

            chromosomes = self._ga._generate_random_population(size, dim, interval)

            if dim == 1:
                population = [chrom[0] for chrom in chromosomes]
            else:
                population = chromosomes

        self._init_diffusion_model(population)

    def run(self, max_generation):
        """
        Starts a diffusion GA. The algorithm performs *max_generation* generations and then stops.
        Old population is completely replaced with a new computed one at the end of each generation.

        Args:
            max_generation (int): Maximum number of GA generations.

        Returns:
            fitness_progress (list): List of average fitness values for each generation (including original population).
        """
        if max_generation < 1:
            raise ValueError('Too few generations...')

        fitness_progress = []
        # we works with numpy arrays in case of diffusion model
        population_size = self._chrom_arr.size

        for generation_num in range(max_generation):
            fitness_sum = numpy.sum(self._fitness_arr)

            fitness_progress.append(fitness_sum / population_size)

            self._chrom_arr, self._fitness_arr = self._compute_diffusion_generation(self._chrom_arr)

        fitness_progress.append(fitness_sum / population_size)

        return fitness_progress