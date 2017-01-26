import pytest
# import standard_ga as ga
from standard_ga import IndividualGA, StandardGA

# @pytest.fixture
# def genalg():
#     return _ga.GeneticAlgorithms()

def fitness_test_func():
    return 1


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
    StandardGA(fitness_test_func(), optim=optim)


def test_init_invalid_optim():
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), optim='LIE!')


@pytest.mark.parametrize('selection', ('rank', 'roulette'))
def test_init_valid_selection_1(selection):
    StandardGA(fitness_test_func(), selection=selection)


def test_init_valid_selection_tournament():
    StandardGA(fitness_test_func(), selection='tournament', tournament_size=2)


def test_init_invalid_selection_type():
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), selection='unknown')


@pytest.mark.parametrize('prob', (0, 1, 0.5))
def test_init_valid_mutation_prob(prob):
    StandardGA(fitness_test_func(), mut_prob=prob)


@pytest.mark.parametrize('prob', (-1, 1.00001, 50))
def test_init_invalid_mutation_prob(prob):
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), mut_prob=prob)


@pytest.mark.parametrize('prob', (0, 1, 0.5))
def test_init_valid_crossover_prob(prob):
    StandardGA(fitness_test_func(), cross_prob=prob)


@pytest.mark.parametrize('prob', (-1, 1.00001, 50))
def test_init_invalid_crossover_prob(prob):
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), cross_prob=prob)


@pytest.mark.parametrize('mut_type', (1, 10, 1000))
def test_init_valid_mut_type(mut_type):
    """
    This function tests only common valid values of mutation type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    StandardGA(fitness_test_func(), mut_type=mut_type)


@pytest.mark.parametrize('mut_type', (-1, 0))
def test_init_invalid_mut_type(mut_type):
    """
    This function tests only common invalid values of mutation type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), mut_type=mut_type)


@pytest.mark.parametrize('cross_type', (1, 10, 1000))
def test_init_valid_cross_type(cross_type):
    """
    This function tests only common valid values of crossover type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    StandardGA(fitness_test_func(), cross_type=cross_type)


@pytest.mark.parametrize('cross_type', (-1, 0))
def test_init_invalid_cross_type(cross_type):
    """
    This function tests only common invalid values of crossover type.
    DO NOT FORGET THAT SUBCLASSES (of StandardGA) HAVE ITS OWN RESTRICTIONS
    THAT MUST BE THOROUGHLY TESTED.
    """
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), cross_type=cross_type)


@pytest.mark.parametrize('elitism', (True, False, 1, 0))
def test_init_valid_elitism(elitism):
    StandardGA(fitness_test_func(), elitism=elitism)


@pytest.mark.parametrize('elitism', (2, -1, 'turn on elitism'))
def test_init_invalid_elitism(elitism):
    with pytest.raises(ValueError):
        StandardGA(fitness_test_func(), elitism=elitism)


def test_best_solution():
    ga = StandardGA(fitness_test_func())
    ga.best_chromosome = [1, 2]
    ga.best_fitness = 155

    assert ga.best_solution == ([1,2], 155)


def test_invalid_random_diff():
    ga = StandardGA(fitness_test_func())

    with pytest.raises(ValueError):
        ga._random_diff(2, 10, start=0)


def test_random_diff_whole_interval():
    ga = StandardGA(fitness_test_func())

    nums = ga._random_diff(5, 5, start=0)

    assert nums == list(range(5))


@pytest.mark.parametrize(['stop', 'n', 'start'], [(6, 5, 0), (50, 49, 0), (100, 99, 0), (1000, 999, 0)])
def test_random_diff_duplicates_and_size(stop, n, start):
    ga = StandardGA(fitness_test_func())

    nums = ga._random_diff(stop, n, start=start)

    assert len(nums) == len(set(nums))
    assert len(nums) == n







