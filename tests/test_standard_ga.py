import pytest

from geneticalgs import StandardGA, RealGA, BinaryGA
from helper_functions import *


def test_individual_ga():
    chromosome = [1,2,3]
    fit_val = 25

    ind_ga = IndividualGA(chromosome, fit_val)

    assert ind_ga.chromosome == chromosome
    assert ind_ga.fitness_val == fit_val


def test_init_fitness_func():
    with pytest.raises(ValueError):
        StandardGA()


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_init_valid_optim(optim):
    StandardGA(fitness_test_sin_func, optim=optim)


def test_init_invalid_optim():
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, optim='LIE!')


@pytest.mark.parametrize('selection', ('rank', 'roulette'))
def test_init_valid_selection_1(selection):
    StandardGA(fitness_test_sin_func, selection=selection)


def test_init_valid_selection_tournament():
    StandardGA(fitness_test_sin_func, selection='tournament', tournament_size=2)


def test_init_invalid_selection_type():
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, selection='unknown')


@pytest.mark.parametrize('prob', (0, 1, 0.5))
def test_init_valid_mutation_prob(prob):
    StandardGA(fitness_test_sin_func, mut_prob=prob)


@pytest.mark.parametrize('prob', (-1, 1.00001, 50))
def test_init_invalid_mutation_prob(prob):
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, mut_prob=prob)


@pytest.mark.parametrize('prob', (0, 1, 0.5))
def test_init_valid_crossover_prob(prob):
    StandardGA(fitness_test_sin_func, cross_prob=prob)


@pytest.mark.parametrize('prob', (-1, 1.00001, 50))
def test_init_invalid_crossover_prob(prob):
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, cross_prob=prob)


@pytest.mark.parametrize('mut_type', (1, 10, 1000))
def test_init_valid_mut_type(mut_type):
    """
    This function tests only common valid values of mutation type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    StandardGA(fitness_test_sin_func, mut_type=mut_type)


@pytest.mark.parametrize('mut_type', (-1, 0))
def test_init_invalid_mut_type(mut_type):
    """
    This function tests only common invalid values of mutation type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, mut_type=mut_type)


@pytest.mark.parametrize('cross_type', (1, 10, 1000))
def test_init_valid_cross_type(cross_type):
    """
    This function tests only common valid values of crossover type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    StandardGA(fitness_test_sin_func, cross_type=cross_type)


@pytest.mark.parametrize('cross_type', (-1, 0))
def test_init_invalid_cross_type(cross_type):
    """
    This function tests only common invalid values of crossover type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, cross_type=cross_type)


@pytest.mark.parametrize('mut_type', (1, 2, 4, len(test_bin_data)))
def test_mutate(mut_type):
    """
    *_mutate()* method is the same (but bit inversion process itself is not)
    for both types of GA: RealGA and Binary GA and thus,
    it is not necessary to test it twice.
    """
    ga = BinaryGA(list(range(10)), fitness_test_func, mut_type=mut_type, mut_prob=1)

    chromosome = []
    mutant = ga._mutate(chromosome)

    assert len(mutant) == mut_type


@pytest.mark.parametrize('cross_type', (1, 2, 4, len(test_bin_data)))
def test_cross(cross_type):
    """
    *_cross()* method is the same (but bit replacement process itself is not)
    for both types of GA: RealGA and Binary GA and thus,
    it is not necessary to test it twice.
    """
    ga = BinaryGA(test_bin_data, fitness_test_func, cross_type=cross_type, cross_prob=1)

    source_chrom = list(range(len(test_bin_data)))
    child = ga._cross([], source_chrom)

    print(child)

    if cross_type > 2:
        assert len(child) == cross_type
    else:
        # single- and two-point crossover differs from other types
        # we don't know the exact replacement interval as its end points are random numbers
        # but this interval must be at least 1
        assert len(child) >= 1


@pytest.mark.parametrize('elitism', (True, False, 1, 0))
def test_init_valid_elitism(elitism):
    StandardGA(fitness_test_sin_func, elitism=elitism)


@pytest.mark.parametrize('elitism', (2, -1, 'turn on elitism'))
def test_init_invalid_elitism(elitism):
    with pytest.raises(ValueError):
        StandardGA(fitness_test_sin_func, elitism=elitism)


def test_best_solution():
    ga = StandardGA(fitness_test_sin_func)
    ga.best_chromosome = [1, 2]
    ga.best_fitness = 155

    assert ga.best_solution == ([1,2], 155)


def test_invalid_random_diff():
    ga = StandardGA(fitness_test_sin_func)

    with pytest.raises(ValueError):
        ga._random_diff(2, 10, start=0)


def test_random_diff_whole_interval():
    ga = StandardGA(fitness_test_sin_func)

    nums = ga._random_diff(5, 5, start=0)

    assert nums == list(range(5))


@pytest.mark.parametrize(['stop', 'n', 'start'], [(6, 5, 0), (50, 49, 0), (100, 99, 0), (1000, 999, 0)])
def test_random_diff_duplicates_and_size(stop, n, start):
    ga = StandardGA(fitness_test_sin_func)

    nums = ga._random_diff(stop, n, start=start)

    assert len(nums) == len(set(nums))
    assert len(nums) == n


@pytest.mark.parametrize(['population', 'size'], [(None, 4), ([], 4), ([1,2,3], 0)])
def test_invalid_conduct_tournament(population, size):
    ga = StandardGA(fitness_test_sin_func)

    with pytest.raises(ValueError):
        ga._conduct_tournament(population, size)


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_conduct_tournament_whole_population(optim):
    ga = StandardGA(fitness_test_sin_func, optim=optim)

    population = list(unsorted_population)
    population = sort_population(optim, population)
    size = len(population)

    if optim == 'max':
        correct_out = (7, 6)
    else:
        correct_out = (1, 2)

    best1, best2 = ga._conduct_tournament(population, size)

    assert (population[best1].fitness_val, population[best2].fitness_val) == correct_out


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_conduct_tournament_population_part(optim):
    ga = StandardGA(fitness_test_sin_func, optim=optim)

    population = list(unsorted_population)
    population = sort_population(optim, population)
    size = 2

    best1, best2 = ga._conduct_tournament(population, size)

    if optim == 'max':
        assert population[best1].fitness_val >= population[best2].fitness_val
    else:
        assert population[best1].fitness_val <= population[best2].fitness_val


@pytest.mark.parametrize(['selection', 'wheel_sum'],
                         [('roulette', None), ('rank', None),
                          ('roulette', 0), ('rank', 0),
                          ('roulette', -1), ('rank', -1)])
def test_select_parents_wheel_sum(selection, wheel_sum):
    ga = StandardGA(fitness_test_sin_func, selection=selection)

    with pytest.raises(ValueError):
        ga._select_parents([], wheel_sum)


def test_select_parents_tournament():
    population = list(unsorted_population)
    population = sort_population('min', population)

    ga = StandardGA(fitness_test_sin_func, optim='min', selection='tournament', tournament_size=len(population))

    parent1, parent2 = ga._select_parents(population)

    assert (parent1.fitness_val, parent2.fitness_val) == (1, 2)


def test_select_parents_unknown_type():
    ga = StandardGA(fitness_test_sin_func)
    ga.selection = 'unknown'

    with pytest.raises(ValueError):
        ga._select_parents([])


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_sort_population(optim):
    ga = StandardGA(fitness_test_sin_func, optim=optim)
    ga.population = list(unsorted_population)
    ga._sort_population()

    required_population = sort_population(optim, list(unsorted_population))

    assert ga.population == required_population


@pytest.mark.parametrize(['chrom', 'fitness', 'optim', 'result'],
                         [(2, 10, 'min', (2, 10)),
                          (2, 101, 'max', (2, 101)),
                          (2, 10, 'max', (1, 100)),
                          (2, 101, 'min', (1, 100))])
def test_update_solution(chrom, fitness, optim, result):
    ga = StandardGA(fitness_test_sin_func, optim=optim)

    ga.best_chromosome = 1
    ga.best_fitness = 100

    ga._update_solution(chrom, fitness)

    assert ga.best_solution == result


@pytest.mark.parametrize(['size', 'result'], [(5, 15), (1000, 500500)])
def test_compute_rank_wheel_sum(size, result):
    ga = StandardGA(fitness_test_sin_func)

    assert ga._compute_rank_wheel_sum(size) == result


def test_extend_population():
    ga = StandardGA(fitness_test_sin_func, optim='min')
    ga.population = [IndividualGA(1, 100)]
    new_elems = [IndividualGA(2, 50), IndividualGA(3, 150)]

    ga.extend_population(new_elems)

    assert ga.best_solution == (2, 50)

    result = []
    for i, individ in zip(range(len(ga.population)), ga.population):
        result.append((individ.chromosome, individ.fitness_val))

    assert result == [(3, 150), (1, 100), (2, 50)]


@pytest.mark.parametrize('chromosomes', (None, [], [1,2]))
def test_invalid_init_population(chromosomes):
    ga = StandardGA(fitness_test_sin_func)

    with pytest.raises(ValueError):
        ga.init_population(chromosomes)


@pytest.mark.parametrize('generations', (0, -1))
def test_invalid_run(generations):
    ga = RealGA(fitness_test_sin_func)

    with pytest.raises(ValueError):
        ga.run(generations)


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_valid_run(optim):
    """
    "Run" function is the same for both types of GA: RealGA and Binary GA and thus,
    it is not necessary to test it twice.
    """
    ga = RealGA(fitness_test_sin_func, optim=optim)
    ga.init_random_population(15, 1, (-5, 5))

    init_best = ga.best_solution

    generations = 10
    fitness_progress = ga.run(generations)

    assert len(fitness_progress) == generations + 1

    if optim == 'min':
        assert init_best[1] >= ga.best_solution[1]
    else:
        assert init_best[1] <= ga.best_solution[1]


