import math


def fitness_test_sin_func(chromosome):
    return math.sin(chromosome)


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


