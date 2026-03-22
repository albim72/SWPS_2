import random
import math

# Funkcja celu: maksimum w pobliżu (3, -2)
def fitness(x, y):
    return -(x - 3)**2 - (y + 2)**2 + 10


# Losowy start
parent = [random.uniform(-10, 10), random.uniform(-10, 10)]
sigma = 0.8

for generation in range(30):
    # Mutacja Gaussowska
    child = [
        parent[0] + random.gauss(0, sigma),
        parent[1] + random.gauss(0, sigma)
    ]

    parent_fit = fitness(parent[0], parent[1])
    child_fit = fitness(child[0], child[1])

    # Selekcja: przeżywa lepszy
    if child_fit > parent_fit:
        parent = child

    print(
        f"Gen {generation:02d} | "
        f"x={parent[0]:7.3f}, y={parent[1]:7.3f}, "
        f"fitness={fitness(parent[0], parent[1]):7.3f}"
    )

print("\nNajlepsze rozwiązanie:", parent)
print("Końcowy fitness:", fitness(parent[0], parent[1]))
