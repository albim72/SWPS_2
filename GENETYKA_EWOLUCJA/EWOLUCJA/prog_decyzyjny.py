import random

# ==========================================================
# DANE TRENINGOWE
# Każdy element to (wartość, prawdziwa_klasa)
# ==========================================================
data = [
    (1.0, 0),
    (1.5, 0),
    (2.0, 0),
    (2.2, 0),
    (2.8, 0),
    (3.0, 1),
    (3.4, 1),
    (3.8, 1),
    (4.2, 1),
    (4.8, 1)
]

# ==========================================================
# FUNKCJA OCENY
# Osobnik = próg decyzyjny
# ==========================================================
def fitness(threshold):
    correct = 0
    for value, true_label in data:
        predicted = 1 if value > threshold else 0
        if predicted == true_label:
            correct += 1
    return correct  # im więcej poprawnych, tym lepiej


# ==========================================================
# PARAMETRY
# ==========================================================
POP_SIZE = 10
GENERATIONS = 20
MUTATION_SCALE = 0.4

# ==========================================================
# INICJALIZACJA
# Losujemy progi z zakresu 0 do 5
# ==========================================================
population = [random.uniform(0, 5) for _ in range(POP_SIZE)]

# ==========================================================
# GŁÓWNA PĘTLA
# ==========================================================
for generation in range(GENERATIONS):
    offspring = []

    for threshold in population:
        child = threshold + random.uniform(-MUTATION_SCALE, MUTATION_SCALE)
        offspring.append(child)

    combined = population + offspring

    # sortujemy malejąco, bo większy fitness jest lepszy
    combined.sort(key=fitness, reverse=True)

    population = combined[:POP_SIZE]

    best = population[0]
    print(
        f"Pokolenie {generation+1}: "
        f"próg = {best:.4f}, poprawnych klasyfikacji = {fitness(best)}"
    )

best = population[0]
print("\nWYNIK KOŃCOWY")
print(f"Najlepszy próg = {best:.4f}")
print(f"Liczba poprawnych klasyfikacji = {fitness(best)}")
