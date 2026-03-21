import random

POP_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.25

DIVISORS = ["2", "3", "4"]
REMAINDERS = ["0", "1", "2", "3"]

data = [
    (1, False),
    (2, True),
    (3, False),
    (4, True),
    (5, False),
    (6, True),
    (7, False),
    (8, True)
]


def random_condition():
    divisor = random.choice(DIVISORS)
    remainder = random.choice(REMAINDERS)
    return f"x % {divisor} == {remainder}"


def fitness(condition):
    correct = 0
    try:
        for x, expected in data:
            result = eval(condition, {"__builtins__": {}}, {"x": x})
            if result == expected:
                correct += 1
        return correct
    except:
        return -999


def mutate(condition):
    parts = condition.split()
    # parts = ['x', '%', '2', '==', '0']

    if random.random() < MUTATION_RATE:
        parts[2] = random.choice(DIVISORS)

    if random.random() < MUTATION_RATE:
        parts[4] = random.choice(REMAINDERS)

    return " ".join(parts)


def crossover(parent1, parent2):
    p1 = parent1.split()
    p2 = parent2.split()

    # dziecko 1 bierze dzielnik od p1, resztę od p2
    child1 = f"x % {p1[2]} == {p2[4]}"

    # dziecko 2 bierze dzielnik od p2, resztę od p1
    child2 = f"x % {p2[2]} == {p1[4]}"

    return child1, child2


def select(population):
    population = sorted(population, key=fitness, reverse=True)
    return population[:2]


population = [random_condition() for _ in range(POP_SIZE)]

for generation in range(GENERATIONS):
    parent1, parent2 = select(population)
    new_population = [parent1, parent2]

    while len(new_population) < POP_SIZE:
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])

    population = new_population[:POP_SIZE]

    best = max(population, key=fitness)
    print(f"Pokolenie {generation+1}: {best} -> fitness = {fitness(best)}")
