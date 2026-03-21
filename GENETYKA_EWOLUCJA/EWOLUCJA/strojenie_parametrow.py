import random

# ==========================================================
# FUNKCJA CELU
# ==========================================================
def f(individual):
    x, y = individual
    return x**2 + y**2


# ==========================================================
# PARAMETRY
# ==========================================================
POP_SIZE = 12
GENERATIONS = 25
MUTATION_SCALE = 0.3

# ==========================================================
# INICJALIZACJA POPULACJI
# Każdy osobnik to lista [x, y]
# ==========================================================
population = [
    [random.uniform(-5, 5), random.uniform(-5, 5)]
    for _ in range(POP_SIZE)
]

# ==========================================================
# GŁÓWNA PĘTLA EWOLUCJI
# ==========================================================
for generation in range(GENERATIONS):
    offspring = []

    for individual in population:
        x, y = individual

        # mutacja obu parametrów
        child = [
            x + random.uniform(-MUTATION_SCALE, MUTATION_SCALE),
            y + random.uniform(-MUTATION_SCALE, MUTATION_SCALE)
        ]
        offspring.append(child)

    # łączymy rodziców i potomków
    combined = population + offspring

    # sortowanie według jakości
    combined.sort(key=f)

    # nowa populacja = najlepsi
    population = combined[:POP_SIZE]

    best = population[0]
    print(
        f"Pokolenie {generation+1}: "
        f"x = {best[0]:.4f}, y = {best[1]:.4f}, f = {f(best):.4f}"
    )

best = population[0]
print("\nWYNIK KOŃCOWY")
print(f"Najlepszy osobnik: x = {best[0]:.4f}, y = {best[1]:.4f}")
print(f"Wartość funkcji = {f(best):.4f}")
