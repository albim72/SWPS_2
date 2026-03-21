import json
import random
import copy

# ==========================================
# PARAMETRY ALGORYTMU EWOLUCYJNEGO
# ==========================================
POP_SIZE = 30
GENERATIONS = 60
MUTATION_SCALE = 0.08
ELITE_SIZE = 6

CRITERIA = ["tech", "communication", "cost", "availability", "fit"]

# ==========================================
# WCZYTANIE DANYCH Z JSON
# ==========================================
with open("candidates.json", "r", encoding="utf-8") as file:
    data = json.load(file)

candidates = data["candidates"]
expert_ranking = data["expert_ranking"]


# ==========================================
# FUNKCJA NORMALIZUJĄCA WAGI
# Suma wag ma być równa 1
# ==========================================
def normalize_weights(weights):
    total = sum(weights.values())
    if total == 0:
        equal = 1.0 / len(weights)
        return {k: equal for k in weights}
    return {k: v / total for k, v in weights.items()}


# ==========================================
# LOSOWY OSOBNIK = LOSOWY ZESTAW WAG
# ==========================================
def create_individual():
    weights = {criterion: random.uniform(0.01, 1.0) for criterion in CRITERIA}
    return normalize_weights(weights)


# ==========================================
# OBLICZANIE WYNIKU KANDYDATA
# ==========================================
def calculate_score(candidate, weights):
    adjusted_cost = 10 - candidate["cost"]  # mniejszy koszt = lepiej

    score = (
        candidate["tech"] * weights["tech"] +
        candidate["communication"] * weights["communication"] +
        adjusted_cost * weights["cost"] +
        candidate["availability"] * weights["availability"] +
        candidate["fit"] * weights["fit"]
    )
    return score


# ==========================================
# TWORZENIE RANKINGU DLA DANYCH WAG
# ==========================================
def build_ranking(candidates, weights):
    scored = []

    for candidate in candidates:
        candidate_copy = candidate.copy()
        candidate_copy["score"] = calculate_score(candidate_copy, weights)
        scored.append(candidate_copy)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [c["name"] for c in scored], scored


# ==========================================
# FUNKCJA FITNESS
# Im mniejsza różnica względem rankingu eksperta,
# tym lepiej
# ==========================================
def fitness(individual, candidates, expert_ranking):
    predicted_ranking, _ = build_ranking(candidates, individual)

    error = 0
    for name in expert_ranking:
        pos_expert = expert_ranking.index(name)
        pos_pred = predicted_ranking.index(name)
        error += abs(pos_expert - pos_pred)

    return -error  # mniejszy błąd -> większy fitness


# ==========================================
# MUTACJA
# Lekko zmieniamy każdą wagę
# ==========================================
def mutate(individual):
    child = copy.deepcopy(individual)

    for key in child:
        child[key] += random.uniform(-MUTATION_SCALE, MUTATION_SCALE)
        if child[key] < 0.001:
            child[key] = 0.001

    return normalize_weights(child)


# ==========================================
# INICJALIZACJA POPULACJI
# ==========================================
population = [create_individual() for _ in range(POP_SIZE)]


# ==========================================
# GŁÓWNA PĘTLA EWOLUCYJNA
# ==========================================
for generation in range(GENERATIONS):
    scored_population = []

    for individual in population:
        fit = fitness(individual, candidates, expert_ranking)
        scored_population.append((individual, fit))

    # sortowanie od najlepszego
    scored_population.sort(key=lambda x: x[1], reverse=True)

    best_individual, best_fit = scored_population[0]

    predicted_names, scored_candidates = build_ranking(candidates, best_individual)

    print(f"\nPokolenie {generation + 1}")
    print(f"Najlepszy fitness: {best_fit}")
    print("Najlepsze wagi:")
    for k, v in best_individual.items():
        print(f"  {k}: {v:.4f}")

    print("Ranking wygenerowany przez najlepszy osobnik:")
    for i, name in enumerate(predicted_names, start=1):
        print(f"  {i}. {name}")

    # elita
    new_population = [ind for ind, _ in scored_population[:ELITE_SIZE]]

    # reszta populacji powstaje przez mutację elit
    while len(new_population) < POP_SIZE:
        parent = random.choice(new_population)
        child = mutate(parent)
        new_population.append(child)

    population = new_population


# ==========================================
# WYNIK KOŃCOWY
# ==========================================
final_scored = [(ind, fitness(ind, candidates, expert_ranking)) for ind in population]
final_scored.sort(key=lambda x: x[1], reverse=True)

best_individual, best_fit = final_scored[0]
final_ranking, final_scored_candidates = build_ranking(candidates, best_individual)

print("\n" + "=" * 50)
print("WYNIK KOŃCOWY")
print("=" * 50)
print(f"Najlepszy fitness: {best_fit}")

print("\nNajlepsze znalezione wagi:")
for k, v in best_individual.items():
    print(f"{k}: {v:.4f}")

print("\nRanking końcowy:")
for i, candidate in enumerate(final_scored_candidates, start=1):
    print(f"{i}. {candidate['name']} -> score = {candidate['score']:.4f}")

print("\nRanking ekspercki:")
for i, name in enumerate(expert_ranking, start=1):
    print(f"{i}. {name}")
