import pytest

from geneticalgs import MigrationGA, RealGA
from helper_functions import *


test_chromosomes = [6, 9, 3, 7, 5, 4, 1, 8, 2]
test_chrom_half_length = len(test_chromosomes) // 2
test_chrom_part1 = test_chromosomes[:test_chrom_half_length]
test_chrom_part2 = test_chromosomes[test_chrom_half_length:]


def test_invalid_init_type():
    with pytest.raises(ValueError):
        MigrationGA(type='wrong_type')


@pytest.mark.parametrize('type', ('binary', 'real'))
def test_valid_init_type(type):
    MigrationGA(type=type)


@pytest.mark.parametrize('ga_list', ([], [1]))
def test_invalid_init_populations(ga_list):
    ga = MigrationGA()

    with pytest.raises(ValueError):
        ga.init_populations(ga_list)


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_valid_init_populations(optim):
    mga = MigrationGA(type='real')

    rga1 = RealGA(fitness_test_sin_func, optim=optim)
    rga2 = RealGA(fitness_test_sin_func, optim=optim)

    size1, size2, dim, interval = 10, 5, 1, (-5, 5)
    rga1.init_random_population(size1, dim, interval)
    rga2.init_random_population(size2, dim, interval)

    mga.init_populations([rga1, rga2])

    assert mga._ga_list_size == 2
    assert len(mga._ga_list) == 2
    assert mga._optim == optim
    assert mga._min_elements == size2

    assert len(mga._ga_list[0].population) == size1
    assert len(mga._ga_list[1].population) == size2

    for rind1, rind2, mind1, mind2 in \
            zip(rga1.population, rga2.population, mga._ga_list[0].population, mga._ga_list[1].population):
        assert rind1.chromosome == mind1.chromosome
        assert rind2.chromosome == mind2.chromosome
        assert rind1.fitness_val == mind1.fitness_val
        assert rind2.fitness_val == mind2.fitness_val


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_compare_solutions(optim):
    mga = MigrationGA(type='real')

    rga1 = RealGA(fitness_test_linear_func, optim=optim)
    rga2 = RealGA(fitness_test_linear_func, optim=optim)

    rga1.init_population(test_chrom_part1, (1, 2))
    rga2.init_population(test_chrom_part2, (1, 2))

    mga.init_populations([rga1, rga2])
    best_solution = mga._compare_solutions()

    if optim == 'min':
        best_chrom = min(test_chromosomes)
    else:
        best_chrom = max(test_chromosomes)

    assert best_solution[0] == best_chrom
    assert best_solution[1] == fitness_test_linear_func(best_chrom)


@pytest.mark.parametrize(['max_generation', 'period', 'migrant_num', 'cloning', 'migrate'],
                         [(0, 0, 3, True, True),
                         (-1, 0, 3, True, True),
                         (1, 2, 3, True, True),
                         (1, 0, 3, True, True),
                         (1, -1, 3, True, True),
                         (50, 10, 8, True, True),  # one population has 5 elements so number of migrants is wrong
                         (50, 10, 4, 'wrong', True),
                         (50, 10, 4, True, 'wrong')
                          ])
def test_invalid_run(max_generation, period, migrant_num, cloning, migrate):
    mga = MigrationGA(type='real')

    rga1 = RealGA(fitness_test_linear_func)
    rga2 = RealGA(fitness_test_linear_func)

    size1, size2, dim, interval = 10, 5, 1, (-5, 5)
    rga1.init_random_population(size1, dim, interval)
    rga2.init_random_population(size2, dim, interval)

    mga.init_populations([rga1, rga2])

    with pytest.raises(ValueError):
        mga.run(max_generation, period, migrant_num, cloning, migrate)


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_valid_run(optim):
    mga = MigrationGA(type='real')

    rga1 = RealGA(fitness_test_linear_func, optim=optim)
    rga2 = RealGA(fitness_test_linear_func, optim=optim)

    rga1.init_population(test_chrom_part1, (0, 100))
    rga2.init_population(test_chrom_part2, (0, 100))

    old_best1 = rga1.best_solution
    old_best2 = rga2.best_solution

    mga.init_populations([rga1, rga2])

    fitness_progress, best_solution = mga.run(10, 5, 1, True)

    assert len(fitness_progress) == 2
    assert len(fitness_progress[0]) == 11
    assert len(fitness_progress[1]) == 11

    if optim == 'min':
        assert best_solution[1] <= old_best1[1]
        assert best_solution[1] <= old_best2[1]
    else:
        assert best_solution[1] >= old_best1[1]
        assert best_solution[1] >= old_best2[1]

