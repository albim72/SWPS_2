import random

def fitness(vec):
    x, y = vec
    return -(x - 5)**2 - (y - 1)**2 + 20


MU = 4       # liczba rodziców
LAMBDA = 12  # liczba potomków
SIGMA = 0.7
GENERATIONS = 25

# populacja początkowa
parents = [
    [random.uniform(-10, 10), random.uniform(-10, 10)]
    for _ in range(MU)
]

for gen in range(GENERATIONS):
    offspring = []

    for _ in range(LAMBDA):
        parent = random.choice(parents)
        child = [
            parent[0] + random.gauss(0, SIGMA),
            parent[1] + random.gauss(0, SIGMA)
        ]
        offspring.append(child)

    # strategia (mu + lambda): rodzice + potomstwo konkurują razem
    combined = parents + offspring
    combined.sort(key=fitness, reverse=True)
    parents = combined[:MU]

    best = parents[0]
    print(
        f"Gen {gen:02d} | "
        f"best=({best[0]:.3f}, {best[1]:.3f}) | "
        f"fitness={fitness(best):.3f}"
    )

print("\nNajlepszy osobnik:", parents[0])
print("Fitness:", fitness(parents[0]))
