import pytest
import numpy
from bitstring import BitArray

from real_ga import RealGA
from standard_ga import IndividualGA
from helper_functions import *


def test_invalid_bin_length():
    ga = RealGA(fitness_test_sin_func)
    ga._bin_length = 128

    with pytest.raises(ValueError):
        ga._check_parameters()


def test_invalid_mut_type():
    with pytest.raises(ValueError):
        RealGA(fitness_test_sin_func, mut_type=1000)


def test_invalid_cross_type():
    with pytest.raises(ValueError):
        RealGA(fitness_test_sin_func, cross_type=1000)


@pytest.mark.parametrize(['bin_length', 'result'], [(32, 9), (64, 12)])
def test_valid_get_mut_bit_offset(bin_length, result):
    ga = RealGA(fitness_test_sin_func)
    ga._bin_length = bin_length

    assert ga._get_mut_bit_offset() == result


def test_invalid_get_mut_bit_offset():
    ga = RealGA(fitness_test_sin_func)
    ga._bin_length = 128

    with pytest.raises(ValueError):
        ga._get_mut_bit_offset()


def test_is_chromosome_list():
    ga = RealGA(fitness_test_sin_func)

    assert not ga._is_chromosome_list(5)
    assert ga._is_chromosome_list([5])
    assert ga._is_chromosome_list([1,2,3,4])


@pytest.mark.parametrize('chromosome', (5, []))
def test_invalid_get_chromosome_return_value(chromosome):
    ga = RealGA(fitness_test_sin_func)

    with pytest.raises(ValueError):
        ga._get_chromosome_return_value(chromosome)


@pytest.mark.parametrize(['chromosome', 'expected'],
                         [([5], 5),
                          ([1, 2], [1, 2]),
                          ([1, 2, 3, 4], [1, 2, 3, 4])
                          ])
def test_valid_get_chromosome_return_value(chromosome, expected):
    ga = RealGA(fitness_test_sin_func)

    assert expected == ga._get_chromosome_return_value(chromosome)


@pytest.mark.parametrize(['var', 'expected'],
                         [(0, 0),
                          (-2, -2),
                          (4, 4),
                          (numpy.nan, 0),
                          (numpy.inf, 5),
                          (-numpy.inf, -3),
                          ([1, 100, -50], [1, 5, -3])
                          ])
def test_adjust_to_interval(var, expected):
    ga = RealGA(fitness_test_sin_func)
    ga.interval = (-3, 5)

    try:
        assert ga._adjust_to_interval(var) == expected
    except ValueError:
        result_arr = ga._adjust_to_interval(var) == expected
        assert result_arr.all()


@pytest.mark.parametrize(['num', 'bit_num'],
                         [(10, [12]),  # 12
                          (456, [14]), # 14
                          (10, [12, 13])
                          ])
def test_invert_bit(num, bit_num):
    ga = RealGA(fitness_test_sin_func, mut_prob=1)
    ga.interval = (num - 1, num + 32)

    original_bstr = BitArray(floatbe=num, length=ga._bin_length).bin

    mutant = ga._invert_bit(num, bit_num)
    result_bstr = BitArray(floatbe=mutant, length=ga._bin_length).bin

    diff = sum(1 for i in range(ga._bin_length) if result_bstr[i] != original_bstr[i] and i in bit_num)

    assert diff == len(bit_num)


@pytest.mark.parametrize(['num', 'start', 'stop', 'bit_idx', 'count'],
                         [(10, 0, 10, [1, 10], 2),
                          (10, 10, 13, [10, 13], 2),
                          (10, 10, 10, [10], 1),
                          (10, 50, 60, [], 0)
                          ])
def test_valid_replace_bits(num, start, stop, bit_idx, count):
    ga = RealGA(fitness_test_sin_func, cross_prob=1)
    ga.interval = (-numpy.inf, numpy.inf)

    source_bstr = BitArray(floatbe=num, length=ga._bin_length).bin

    new_chromosome = ga._replace_bits(num, 0, start, stop)
    actual_bstr = BitArray(floatbe=new_chromosome, length=ga._bin_length).bin

    for i in bit_idx:
        assert actual_bstr[i] == source_bstr[i]

    assert actual_bstr.count('1') == count


@pytest.mark.parametrize(['start', 'stop'],
                         [(10, 0),
                          (-1, 50),
                          (-7, -2),
                          (32, 64)
                          ])
def test_invalid_replace_bits(start, stop):
    ga = RealGA(fitness_test_sin_func)

    with pytest.raises(ValueError):
        ga._replace_bits(1, 0, start, stop)


@pytest.mark.parametrize('chromosome', (1, -2, 5, -654, 1087))
def test_compute_fitness(chromosome):
    ga = RealGA(fitness_test_linear_func)

    assert ga._compute_fitness(chromosome) == 2 * chromosome  # 2*x is linear test function


@pytest.mark.parametrize(['size', 'dim', 'interval'],
                         [(None, 4, (1, 2)),
                          (5, None, (1, 2)),
                          (4, 7, None),
                          (0, 5, (1, 2)),
                          (1, 5, (1, 2)),
                          (-5, 4, (1, 2)),
                          (4, 0, (1, 2)),
                          (4, -2, (1, 2)),
                          (4, 2, (3, 2)),
                          (4, 2, (3, 3)),
                          ])
def test_invalid_check_init_random_population(size, dim, interval):
    ga = RealGA(fitness_test_linear_func)

    with pytest.raises(ValueError):
        ga._check_init_random_population(size, dim, interval)


def test_valid_check_init_random_population():
    size, dim, interval = 2, 3, (-5, 5)

    ga = RealGA(fitness_test_linear_func)

    ga._check_init_random_population(size, dim, interval)


@pytest.mark.parametrize(['size', 'dim', 'interval'],
                         [(300, 4, (-5, 5)),
                          (5000, 1, (100.3, 500.5)),
                         ])
def test_generate_random_population(size, dim, interval):
    ga = RealGA(fitness_test_linear_func)

    population = ga._generate_random_population(size, dim, interval)
    res_arr0 = population >= interval[0]
    res_arr1 = population < interval[1]

    assert len(population) == size
    assert len(population[0]) == dim
    assert res_arr0.all()
    assert res_arr1.all()


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_init_random_population(optim):
    ga = RealGA(fitness_test_linear_func, optim=optim)

    size, dim, interval = 3, 1, (-5, 5)
    ga.init_random_population(size, dim, interval)

    assert len(ga.population) == size
    assert ga.population[size - 1].fitness_val == ga.best_solution[1]
    assert ga.population[size - 1].chromosome == ga.best_solution[0]

    if optim == 'max':
        for i in range(size - 1):
            assert ga.population[i].fitness_val <= ga.best_solution[1]
            assert ga.population[0].fitness_val <= ga.population[i].fitness_val
    else:
        for i in range(size - 1):
            assert ga.population[i].fitness_val >= ga.best_solution[1]
            assert ga.population[0].fitness_val >= ga.population[i].fitness_val


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_valid_init_population(optim):
    chromosomes = list(test_real_chromosomes)
    expected_population = []
    for chromosome in chromosomes:
        expected_population.append(IndividualGA(chromosome, fitness_test_sin_func(chromosome)))

    expected_population = sort_population(optim, expected_population)

    ga = RealGA(fitness_test_sin_func, optim=optim)
    interval = (-10, 10)
    ga.init_population(test_real_chromosomes, interval)

    assert ga.interval == interval

    best_solution = ga.best_solution
    if optim == 'min':
        assert test_real_best_min_ind[0] == best_solution[0]
    else:
        assert test_real_best_max_ind[0] == best_solution[0]

    for actual, expected in zip(ga.population, expected_population):
        assert actual.chromosome == expected.chromosome


def test_invalid_init_population():
    optim = 'min'

    chromosomes = list(test_real_chromosomes)
    expected_population = []
    for chromosome in chromosomes:
        expected_population.append(IndividualGA(chromosome, fitness_test_sin_func(chromosome)))

    expected_population = sort_population(optim, expected_population)

    ga = RealGA(fitness_test_sin_func, optim=optim)
    interval = (10, 4)

    with pytest.raises(ValueError):
        ga.init_population(test_real_chromosomes, interval)


