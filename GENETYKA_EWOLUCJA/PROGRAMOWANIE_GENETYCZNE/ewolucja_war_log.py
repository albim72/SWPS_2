import random

POP_SIZE = 10
GENERATIONS = 15
MUTATION_RATE = 0.25

OPERATORS = [">", "<", ">=", "<=", "=="]
VALUES = ["1", "2", "3", "4", "5", "6"]

data = [
    (1, False),
    (2, False),
    (3, False),
    (4, True),
    (5, True),
    (6, True)
]


def random_condition():
    op = random.choice(OPERATORS)
    value = random.choice(VALUES)
    return f"x {op} {value}"


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
    if random.random() < MUTATION_RATE:
        parts[1] = random.choice(OPERATORS)
    if random.random() < MUTATION_RATE:
        parts[2] = random.choice(VALUES)
    return " ".join(parts)


def crossover(parent1, parent2):
    p1 = parent1.split()
    p2 = parent2.split()

    child1 = f"x {p1[1]} {p2[2]}"
    child2 = f"x {p2[1]} {p1[2]}"
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
