import random
import math

def fitness(x, y):
    return -(x - 2)**2 - (y - 6)**2 + 15


# osobnik: [x, y, sigma]
parent = [random.uniform(-10, 10), random.uniform(-10, 10), 1.0]

for gen in range(35):
    x, y, sigma = parent

    # samoadaptacja sigma
    new_sigma = sigma * math.exp(random.gauss(0, 0.15))
    new_sigma = max(0.01, min(3.0, new_sigma))

    child = [
        x + random.gauss(0, new_sigma),
        y + random.gauss(0, new_sigma),
        new_sigma
    ]

    parent_fit = fitness(parent[0], parent[1])
    child_fit = fitness(child[0], child[1])

    if child_fit > parent_fit:
        parent = child

    print(
        f"Gen {gen:02d} | "
        f"x={parent[0]:7.3f}, y={parent[1]:7.3f}, "
        f"sigma={parent[2]:6.3f}, "
        f"fitness={fitness(parent[0], parent[1]):7.3f}"
    )

print("\nNajlepsze rozwiązanie:")
print(f"x={parent[0]:.3f}, y={parent[1]:.3f}, sigma={parent[2]:.3f}")
print("Fitness:", fitness(parent[0], parent[1]))
