import random

# ==========================================================
# FUNKCJA CELU
# ==========================================================
def f(x):
    return x**2 + 4*x + 5


# ==========================================================
# PARAMETRY ALGORYTMU
# ==========================================================
POP_SIZE = 10
GENERATIONS = 20
MUTATION_SCALE = 0.5

# ==========================================================
# INICJALIZACJA POPULACJI
# Każdy osobnik to jedna liczba rzeczywista x
# ==========================================================
population = [random.uniform(-10, 10) for _ in range(POP_SIZE)]


# ==========================================================
# GŁÓWNA PĘTLA EWOLUCJI
# ==========================================================
for generation in range(GENERATIONS):
    offspring = []

    # Tworzenie potomków przez mutację
    for x in population:
        child = x + random.uniform(-MUTATION_SCALE, MUTATION_SCALE)
        offspring.append(child)

    # Łączymy rodziców i potomków
    combined = population + offspring

    # Sortujemy według jakości: im mniejsza wartość f(x), tym lepiej
    combined.sort(key=f)

    # Wybieramy najlepsze osobniki do nowej populacji
    population = combined[:POP_SIZE]

    # Najlepszy osobnik w aktualnym pokoleniu
    best = population[0]
    print(f"Pokolenie {generation+1}: najlepsze x = {best:.4f}, f(x) = {f(best):.4f}")

# Wynik końcowy
best = population[0]
print("\nWYNIK KOŃCOWY")
print(f"Najlepsze znalezione x = {best:.4f}")
print(f"Wartość funkcji = {f(best):.4f}")
