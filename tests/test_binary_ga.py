import pytest

from binary_ga import BinaryGA


test_data = [-5, -4, -3, 3, 4, 5, 10]
test_best_min_ind = ([0, 1, 2], -12)
test_best_max_ind = ([3, 4, 5, 6], 12)


def fitness_test_func(chromosome, data):
    result_sum = 0

    for bit in chromosome:
        result_sum += data[bit]

    return result_sum


@pytest.mark.parametrize('data', (None, [], [1, 2, 3]))
def test_invalid_init_data(data):
    with pytest.raises(ValueError):
        BinaryGA(data, fitness_test_func)


def test_invalid_mut_type():
    with pytest.raises(ValueError):
        BinaryGA(list(range(5)), fitness_test_func, mut_type=10)


def test_invalid_cross_type():
    with pytest.raises(ValueError):
        BinaryGA(list(range(5)), fitness_test_func, cross_type=10)


def test_invert_bit():
    ga = BinaryGA(test_data, fitness_test_func, mut_prob=1)

    chromosome = [10, 5, 20]
    bit_to_invert = [10, 7, 9]

    mutant = ga._invert_bit(chromosome, bit_to_invert)

    assert len(mutant) == 4
    assert 10 not in mutant
    assert 5 in mutant
    assert 20 in mutant
    assert 7 in mutant
    assert 9 in mutant


@pytest.mark.parametrize(['start', 'stop', 'result'],
                         [(6, 9, []),
                          (10, 10, [10]),
                          (1, 15, [5, 10])
                          ])
def test_valid_replace_bits(start, stop, result):
    ga = BinaryGA(list(range(21)), fitness_test_func, cross_prob=1)

    source = [10, 5, 20]
    mutant = ga._replace_bits(source, [], start, stop)

    assert len(mutant) == len(result)

    for bit in result:
        assert bit in mutant


@pytest.mark.parametrize(['start', 'stop'],
                         [(6, 2),
                          (-1, 6),
                          (5, -1),
                          (5, 15)
                          ])
def test_invalid_replace_bits(start, stop):
    ga = BinaryGA(list(range(10)), fitness_test_func)

    with pytest.raises(ValueError):
        ga._replace_bits([], [], start, stop)


@pytest.mark.parametrize(['chromosome', 'expected'],
                         [([2, 5], 2),
                          ([6], 10),
                          ])
def test_compute_fitness(chromosome, expected):
    ga = BinaryGA(test_data, fitness_test_func)

    assert ga._compute_fitness(chromosome) == expected


@pytest.mark.parametrize(['number', 'result'],
                         [(3, [9, 8]),
                          (0, []),
                          (4, [7]),
                          (534, [8, 7, 5, 0])
                          ])
def test_valid_get_bit_positions(number, result):
    ga = BinaryGA(list(range(10)), fitness_test_func)

    bin_repr = ga._get_bit_positions(number)

    assert len(bin_repr) == len(result)

    for bit in result:
        assert bit in bin_repr


def test_invalid_get_bit_positions():
    ga = BinaryGA(list(range(10)), fitness_test_func)

    with pytest.raises(ValueError):
        ga._get_bit_positions(-1)


def test_valid_check_init_random_population():
    ga = BinaryGA(list(range(5)), fitness_test_func)

    max_num = ga._check_init_random_population(15)
    assert max_num == 2**5


@pytest.mark.parametrize('size', (None, 3, -1, 0, 32, 100))
def test_invalid_check_init_random_population(size):
    ga = BinaryGA(list(range(5)), fitness_test_func)

    with pytest.raises(ValueError):
        ga._check_init_random_population(size)


@pytest.mark.parametrize('size', (5, 20))
def test_generate_random_population(size):
    ga = BinaryGA(list(range(5)), fitness_test_func)
    max_num = 2**5

    population = ga._generate_random_population(max_num, size)

    assert len(population) == size
    assert len(population) == len(set(population))

    for num in population:
        assert max_num > num >= 1


@pytest.mark.parametrize('optim', ('min', 'max'))
def test_init_random_population(optim):
    ga = BinaryGA(test_data, fitness_test_func, optim=optim)

    pop_size = 5
    ga.init_random_population(pop_size)

    assert len(ga.population) == pop_size
    assert ga.population[pop_size - 1].fitness_val == ga.best_solution[1]
    assert ga.population[pop_size - 1].chromosome == ga.best_solution[0]

    if optim == 'max':
        for i in range(pop_size - 1):
            assert ga.population[i].fitness_val <= ga.best_solution[1]
            assert ga.population[0].fitness_val <= ga.population[i].fitness_val
    else:
        for i in range(pop_size - 1):
            assert ga.population[i].fitness_val >= ga.best_solution[1]
            assert ga.population[0].fitness_val >= ga.population[i].fitness_val


