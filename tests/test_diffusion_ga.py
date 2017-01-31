import pytest
import numpy
import random

from geneticalgs import DiffusionGA, RealGA, BinaryGA
from helper_functions import *


def test_get_population():
    ga = DiffusionGA('just_for_test')

    arr = numpy.empty(4)
    for i in range(4):
        arr[i] = i

    ga._fitness_arr = numpy.empty(2)
    ga._fitness_arr[0] = arr[0]
    ga._fitness_arr[1] = arr[1]
    ga._chrom_arr = numpy.empty(2)
    ga._chrom_arr[0] = arr[2]
    ga._chrom_arr[1] = arr[3]

    assert (ga.population[0] == arr[2:]).all()
    assert (ga.population[1] == arr[:2]).all()


@pytest.mark.parametrize(['fit_list', 'row', 'column'],
                         [(random.sample(range(100), 9), 0, 0),
                          (random.sample(range(100), 9), 2, 2),
                          (random.sample(range(100), 9), 0, 2),
                          (random.sample(range(100), 9), 2, 0),
                          (random.sample(range(100), 9), 1, 1),
                          (random.sample(range(100), 9), 0, 1),
                          (random.sample(range(100), 9), 1, 0),
                          (random.sample(range(100), 9), 1, 2),
                          (random.sample(range(100), 9), 2, 1)
                          ])
def test_get_neighbour(fit_list, row, column):
    ga = RealGA(fitness_test_sin_func)
    dga = DiffusionGA(ga)
    dga._chrom_arr = numpy.array(list(range(9))).reshape((3, 3))
    dga._fitness_arr = numpy.array(fit_list).reshape(3, 3)
    shape = dga._chrom_arr.shape

    selected_chromosome = dga._get_neighbour(row, column)

    indices = numpy.where(dga._chrom_arr == selected_chromosome)
    coords = (indices[0][0], indices[1][0])

    valid_indices = [((row - 1) % shape[0], column),
                     ((row + 1) % shape[0], column),
                     (row, (column - 1) % shape[1]),
                     (row, (column + 1) % shape[1])]

    assert coords in valid_indices


def test_invalid_find_critical_values():
    dga = DiffusionGA('just_for_test')

    fit_arr = [
        [[1,2,4], [1,2,4]],
        [[2,6,8], [1,2,4]],
        [[1,2,4], [8,7,5]]
    ]
    fit_arr = numpy.array(fit_arr)

    with pytest.raises(ValueError):
        dga._find_critical_values(fit_arr)


@pytest.mark.parametrize('optim', ('min', 'max'))
@pytest.mark.parametrize(['arr', 'rmin', 'rmax'],
                         [([1,2,-1,9,2], -1, 9),  # 1D
                          ([[45,4,-78,8,4], [-3,55,35,55,-35]], -78, 55)  # 2D
                         ])
def test_valid_find_critical_values(optim, arr, rmin, rmax):
    ga = RealGA(fitness_test_sin_func, optim=optim)
    dga = DiffusionGA(ga)

    arr = numpy.array(arr)

    coords_best, coords_worst = dga._find_critical_values(arr)

    if optim == 'min':
        assert arr[coords_best] == rmin
        assert arr[coords_worst] == rmax
    else:
        assert arr[coords_best] == rmax
        assert arr[coords_worst] == rmin


@pytest.mark.parametrize('population', (None, [], [1,2,3]))
def test_invalid_init_population(population):
    with pytest.raises(ValueError):
        DiffusionGA('just_for_test').init_population(population)


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_valid_init_population(optim):
    """
    This test includes testing of the following parts: *init_population()*,
    *_init_diffusion_model()*, *_construct_diffusion_model()* and *best_solution*.
    """
    ga = RealGA(fitness_test_linear_func, optim=optim)
    dga = DiffusionGA(ga)

    array_side = 3
    population = list(range(array_side * array_side))
    fitness = [fitness_test_linear_func(chrom) for chrom in population]

    dga.init_population(population)

    for elem, fit in zip(population, fitness):
        assert elem in dga._chrom_arr
        assert fit in dga._fitness_arr

    if optim == 'min':
        assert dga.best_solution[0] == min(population)
        assert dga.best_solution[1] == fitness_test_linear_func(min(population))
    else:
        assert dga.best_solution[0] == max(population)
        assert dga.best_solution[1] == fitness_test_linear_func(max(population))


@pytest.mark.parametrize(['size', 'dim', 'interval', 'ga_inst'],
                         [(10, None, None, RealGA(fitness_test_linear_func)),
                         (None, None, None, RealGA(fitness_test_linear_func)),
                         (10, 3, None, RealGA(fitness_test_linear_func)),
                         (None, None, None, BinaryGA([1,2,3,4], fitness_test_linear_func))
                          ])
def test_invalid_init_random_population(size, dim, interval, ga_inst):
    with pytest.raises(ValueError):
        DiffusionGA(ga_inst).init_random_population(size, dim, interval)


@pytest.mark.parametrize('optim', ('min', 'max'))
@pytest.mark.parametrize('type', ('real', 'binary'))
def test_init_random_population_real(optim, type):
    size, dim, interval = 9, 1, (-5, 5)

    if type == 'real':
        ga = RealGA(fitness_test_linear_func, optim=optim)
        dga = DiffusionGA(ga)

        dga.init_random_population(size, dim, interval)
    else:
        ga = BinaryGA(test_bin_data, fitness_test_func, optim=optim)
        dga = DiffusionGA(ga)
        dga.init_random_population(size)

    assert dga.population[0].size == size
    assert dga.population[1].size == size

    shape = dga.population[0].shape
    if optim == 'max':
        for row in range(shape[0]):
            for column in range(shape[1]):
                assert dga.population[1][row][column] <= dga.best_solution[1]
    else:
        for row in range(shape[0]):
            for column in range(shape[1]):
                assert dga.population[1][row][column] >= dga.best_solution[1]


@pytest.mark.parametrize('generations', (0, -1))
def test_invalid_run(generations):
    with pytest.raises(ValueError):
        DiffusionGA('just_for_test').run(generations)


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_valid_run(optim):
    """
    "Run" function is the same for both types of diffusion GA: RealGA and Binary GA and thus,
    it is not necessary to test it twice.
    """
    ga = RealGA(fitness_test_sin_func, optim=optim)
    dga = DiffusionGA(ga)
    dga.init_random_population(11, 1, (-5, 5))  # size is 11 but actually wil be 9 (truncated square root from 11)

    init_best = dga.best_solution

    generations = 10
    fitness_progress = dga.run(generations)

    assert len(fitness_progress) == generations + 1

    if optim == 'min':
        assert init_best[1] >= dga.best_solution[1]
    else:
        assert init_best[1] <= dga.best_solution[1]








