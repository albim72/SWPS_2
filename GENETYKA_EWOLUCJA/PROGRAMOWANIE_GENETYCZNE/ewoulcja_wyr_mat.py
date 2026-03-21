import random

POP_SIZE = 8
GENERATIONS = 12
MUTATION_RATE = 0.3

VARIABLES = ["x"]
CONSTANTS = ["1", "2", "3", "4", "5"]
OPERATORS = ["+", "-", "*"]


def random_atom():
    return random.choice(VARIABLES + CONSTANTS)


def random_expression():
    # budujemy proste wyrażenie typu: a op b
    left = random_atom()
    op = random.choice(OPERATORS)
    right = random_atom()
    return f"{left} {op} {right}"


def fitness(expr):
    x = 3
    target = 10
    try:
        value = eval(expr, {"__builtins__": {}}, {"x": x})
        return -abs(target - value)
    except:
        return -999


def mutate(expr):
    parts = expr.split()
    if random.random() < MUTATION_RATE:
        parts[0] = random_atom()
    if random.random() < MUTATION_RATE:
        parts[1] = random.choice(OPERATORS)
    if random.random() < MUTATION_RATE:
        parts[2] = random_atom()
    return " ".join(parts)


def crossover(parent1, parent2):
    p1 = parent1.split()
    p2 = parent2.split()

    child1 = f"{p1[0]} {p1[1]} {p2[2]}"
    child2 = f"{p2[0]} {p2[1]} {p1[2]}"
    return child1, child2


def select(population):
    population = sorted(population, key=fitness, reverse=True)
    return population[:2]


population = [random_expression() for _ in range(POP_SIZE)]

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
