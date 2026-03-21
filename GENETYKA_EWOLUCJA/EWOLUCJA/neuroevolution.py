import json
import random
import math
import copy

# ==========================================
# PARAMETRY ALGORYTMU
# ==========================================
POP_SIZE = 40
GENERATIONS = 80
MUTATION_SCALE = 0.15
ELITE_SIZE = 8

INPUT_SIZE = 5
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1

# liczba parametrów sieci:
# W1: 5*4 = 20
# b1: 4
# W2: 4*1 = 4
# b2: 1
GENOME_LENGTH = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE


# ==========================================
# WCZYTANIE DANYCH
# ==========================================
with open("candidates.json", "r", encoding="utf-8") as file:
    data = json.load(file)

candidates = data["candidates"]
expert_ranking = data["expert_ranking"]


# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def relu(x):
    return max(0.0, x)


def candidate_to_features(candidate):
    adjusted_cost = 10 - candidate["cost"]
    return [
        float(candidate["tech"]),
        float(candidate["communication"]),
        float(adjusted_cost),
        float(candidate["availability"]),
        float(candidate["fit"])
    ]


# ==========================================
# REPREZENTACJA OSOBNIKA
# Osobnik = lista liczb = wszystkie parametry sieci
# ==========================================
def create_individual():
    return [random.uniform(-1.0, 1.0) for _ in range(GENOME_LENGTH)]


def decode_individual(genome):
    idx = 0

    # W1: INPUT_SIZE x HIDDEN_SIZE
    W1 = []
    for _ in range(INPUT_SIZE):
        row = []
        for _ in range(HIDDEN_SIZE):
            row.append(genome[idx])
            idx += 1
        W1.append(row)

    # b1: HIDDEN_SIZE
    b1 = []
    for _ in range(HIDDEN_SIZE):
        b1.append(genome[idx])
        idx += 1

    # W2: HIDDEN_SIZE x OUTPUT_SIZE (czyli 4 x 1)
    W2 = []
    for _ in range(HIDDEN_SIZE):
        row = []
        for _ in range(OUTPUT_SIZE):
            row.append(genome[idx])
            idx += 1
        W2.append(row)

    # b2: OUTPUT_SIZE
    b2 = []
    for _ in range(OUTPUT_SIZE):
        b2.append(genome[idx])
        idx += 1

    return W1, b1, W2, b2


# ==========================================
# FORWARD PASS SIECI
# ==========================================
def neural_score(features, genome):
    W1, b1, W2, b2 = decode_individual(genome)

    # warstwa ukryta
    hidden = []
    for j in range(HIDDEN_SIZE):
        s = 0.0
        for i in range(INPUT_SIZE):
            s += features[i] * W1[i][j]
        s += b1[j]
        hidden.append(relu(s))

    # warstwa wyjściowa
    output = 0.0
    for j in range(HIDDEN_SIZE):
        output += hidden[j] * W2[j][0]
    output += b2[0]

    return output


# ==========================================
# BUDOWANIE RANKINGU
# ==========================================
def build_ranking(candidates, genome):
    scored = []

    for candidate in candidates:
        features = candidate_to_features(candidate)
        score = neural_score(features, genome)

        candidate_copy = candidate.copy()
        candidate_copy["score"] = score
        scored.append(candidate_copy)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [c["name"] for c in scored], scored


# ==========================================
# FITNESS
# ==========================================
def fitness(genome, candidates, expert_ranking):
    predicted_ranking, _ = build_ranking(candidates, genome)

    error = 0
    for name in expert_ranking:
        pos_expert = expert_ranking.index(name)
        pos_pred = predicted_ranking.index(name)
        error += abs(pos_expert - pos_pred)

    return -error


# ==========================================
# MUTACJA
# ==========================================
def mutate(genome):
    child = copy.deepcopy(genome)

    for i in range(len(child)):
        child[i] += random.uniform(-MUTATION_SCALE, MUTATION_SCALE)

    return child


# ==========================================
# INICJALIZACJA POPULACJI
# ==========================================
population = [create_individual() for _ in range(POP_SIZE)]


# ==========================================
# GŁÓWNA PĘTLA EWOLUCYJNA
# ==========================================
for generation in range(GENERATIONS):
    scored_population = []

    for genome in population:
        fit = fitness(genome, candidates, expert_ranking)
        scored_population.append((genome, fit))

    scored_population.sort(key=lambda x: x[1], reverse=True)

    best_genome, best_fit = scored_population[0]
    predicted_names, scored_candidates = build_ranking(candidates, best_genome)

    print(f"\nPokolenie {generation + 1}")
    print(f"Najlepszy fitness: {best_fit}")
    print("Ranking wygenerowany przez najlepszą sieć:")
    for i, name in enumerate(predicted_names, start=1):
        print(f"  {i}. {name}")

    # elita
    new_population = [genome for genome, _ in scored_population[:ELITE_SIZE]]

    # reszta przez mutację elit
    while len(new_population) < POP_SIZE:
        parent = random.choice(new_population)
        child = mutate(parent)
        new_population.append(child)

    population = new_population


# ==========================================
# WYNIK KOŃCOWY
# ==========================================
final_scored = [(genome, fitness(genome, candidates, expert_ranking)) for genome in population]
final_scored.sort(key=lambda x: x[1], reverse=True)

best_genome, best_fit = final_scored[0]
final_ranking, final_scored_candidates = build_ranking(candidates, best_genome)

print("\n" + "=" * 50)
print("WYNIK KOŃCOWY")
print("=" * 50)
print(f"Najlepszy fitness: {best_fit}")

print("\nRanking końcowy:")
for i, candidate in enumerate(final_scored_candidates, start=1):
    print(f"{i}. {candidate['name']} -> score = {candidate['score']:.4f}")

print("\nRanking ekspercki:")
for i, name in enumerate(expert_ranking, start=1):
    print(f"{i}. {name}")
