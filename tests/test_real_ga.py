import pytest
import math
import numpy
from bitstring import BitArray

from real_ga import RealGA
from standard_ga import IndividualGA


test_chromosomes = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
test_best_min_ind = (4.0, -0.7568024953079282)
test_best_max_ind = (1.5, 0.9974949866040544)

unsorted_population = [IndividualGA(1, 3), IndividualGA(1, 1), IndividualGA(1, 2), IndividualGA(1, 7), IndividualGA(1, 6)]


def fitness_test_func(chromosome):
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


def test_invalid_bin_length():
    ga = RealGA(fitness_test_func)
    ga._bin_length = 128

    with pytest.raises(ValueError):
        ga._check_parameters()


def test_invalid_mut_type():
    with pytest.raises(ValueError):
        RealGA(fitness_test_func, mut_type=1000)


def test_invalid_cross_type():
    with pytest.raises(ValueError):
        RealGA(fitness_test_func, cross_type=1000)


@pytest.mark.parametrize(['bin_length', 'result'], [(32, 9), (64, 12)])
def test_valid_get_mut_bit_offset(bin_length, result):
    ga = RealGA(fitness_test_func)
    ga._bin_length = bin_length

    assert ga._get_mut_bit_offset() == result


def test_invalid_get_mut_bit_offset():
    ga = RealGA(fitness_test_func)
    ga._bin_length = 128

    with pytest.raises(ValueError):
        ga._get_mut_bit_offset()


def test_is_chromosome_list():
    ga = RealGA(fitness_test_func)

    assert not ga._is_chromosome_list(5)
    assert ga._is_chromosome_list([5])
    assert ga._is_chromosome_list([1,2,3,4])


@pytest.mark.parametrize('chromosome', (5, []))
def test_invalid_get_chromosome_return_value(chromosome):
    ga = RealGA(fitness_test_func)

    with pytest.raises(ValueError):
        ga._get_chromosome_return_value(chromosome)


@pytest.mark.parametrize(['chromosome', 'expected'],
                         [([5], 5),
                          ([1, 2], [1, 2]),
                          ([1, 2, 3, 4], [1, 2, 3, 4])
                          ])
def test_valid_get_chromosome_return_value(chromosome, expected):
    ga = RealGA(fitness_test_func)

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
    ga = RealGA(fitness_test_func)
    ga.interval = (-3, 5)

    try:
        assert ga._adjust_to_interval(var) == expected
    except ValueError:
        result_arr = ga._adjust_to_interval(var) == expected
        assert result_arr.all()


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_init_population(optim):
    chromosomes = list(test_chromosomes)
    expected_population = []
    for chromosome in chromosomes:
        expected_population.append(IndividualGA(chromosome, fitness_test_func(chromosome)))

    expected_population = sort_population(optim, expected_population)

    ga = RealGA(fitness_test_func, optim=optim)
    ga.init_population(test_chromosomes)

    best_solution = ga.best_solution

    if optim == 'min':
        assert test_best_min_ind[0] == best_solution[0]
    else:
        assert test_best_max_ind[0] == best_solution[0]

    for actual, expected in zip(ga.population, expected_population):
        assert actual.chromosome == expected.chromosome


@pytest.mark.parametrize(['num', 'bit_num'],
                         [(10, [12]),  # 12
                          (456, [14]), # 14
                          (10, [12, 13])
                          ])
def test_invert_bit(num, bit_num):
    ga = RealGA(fitness_test_func, mut_prob=1)
    ga.interval = (num - 1, num + 32)

    original_bstr = BitArray(floatbe=num, length=ga._bin_length).bin

    mutant = ga._invert_bit(num, bit_num)
    result_bstr = BitArray(floatbe=mutant, length=ga._bin_length).bin

    diff = sum(1 for i in range(ga._bin_length) if result_bstr[i] != original_bstr[i])

    assert diff == len(bit_num)


# @pytest.mark.parametrize(['num', 'bit_num'],
#                          [(10, [12]),  # 12
#                           (456, [14]), # 14
#                           (10, [12, 13])
#                           ])
# def test_replace_bits(num, bit_num):
#     ga = RealGA(fitness_test_func, cross_prob=1)
#     ga.interval = (num - 1, num + 32)
#
#     original_bstr = BitArray(floatbe=num, length=ga._bin_length).bin
#
#     mutant = ga._invert_bit(num, bit_num)
#     result_bstr = BitArray(floatbe=mutant, length=ga._bin_length).bin
#
#     diff = sum(1 for i in range(ga._bin_length) if result_bstr[i] != original_bstr[i])
#
#     assert diff == len(bit_num)







