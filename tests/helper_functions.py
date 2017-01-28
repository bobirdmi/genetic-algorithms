import math
from standard_ga import IndividualGA


# BinaryGA
test_bin_data = [-5, -4, -3, 3, 4, 5, 10]
test_bin_best_min_ind = ([0, 1, 2], -12)
test_bin_best_max_ind = ([3, 4, 5, 6], 12)
test_bin_population = [[0, 3], [1, 4, 5], [2, 4, 0], [3, 4, 5, 6]]
test_bin_population_best_min = ([2, 4, 0], -4)
test_bin_population_best_max = ([3, 4, 5, 6], 22)

# RealGA
test_real_chromosomes = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
test_real_best_min_ind = (4.0, -0.7568024953079282)
test_real_best_max_ind = (1.5, 0.9974949866040544)

# common
unsorted_population = [IndividualGA(1, 3), IndividualGA(1, 1), IndividualGA(1, 2), IndividualGA(1, 7), IndividualGA(1, 6)]


def fitness_test_func(chromosome, data):
    # only for BinaryGA
    result_sum = 0

    for bit in chromosome:
        result_sum += data[bit]

    return result_sum


def fitness_test_sin_func(chromosome):
    return math.sin(chromosome)


def fitness_test_linear_func(chromosome):
    return chromosome*2


def sort_population(optim, population):
    """
    Sorts population if IndividualGA objects.
    """
    if optim == 'max':
        # an algorithm maximizes a fitness value
        # ascending order
        population.sort(key=lambda x: x.fitness_val)
    else:
        # an algorithm minimizes a fitness value
        # descending order
        population.sort(key=lambda x: x.fitness_val, reverse=True)

    return population


