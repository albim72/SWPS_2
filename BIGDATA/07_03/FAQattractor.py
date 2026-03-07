from math import sqrt


class ResonantAttractor:
    """
    Atraktor rezonansowy:
    nie wybiera maksimum jednej cechy,
    lecz szuka idei najbliższej wzorcowi znaczenia.
    """

    def __init__(self, target_vector: list[float], feature_names: list[str]):
        self.target_vector = target_vector
        self.feature_names = feature_names

    def distance(self, vector: list[float]) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(vector, self.target_vector)))

    def resonance(self, vector: list[float]) -> float:
        return 1 / (1 + self.distance(vector))

    def select_best(self, items: list[dict]) -> dict:
        return max(items, key=lambda item: self.resonance(item["vector"]))

    def rank(self, items: list[dict]) -> list[dict]:
        ranked = []
        for item in items:
            score = self.resonance(item["vector"])
            ranked.append({
                "name": item["name"],
                "vector": item["vector"],
                "resonance": score
            })
        return sorted(ranked, key=lambda x: x["resonance"], reverse=True)


ideas = [
    {
        "name": "AutoRaport AI dla firm",
        "vector": [0.65, 0.95, 0.80, 0.60, 0.40]
    },
    {
        "name": "SX Decision Layer",
        "vector": [0.95, 0.70, 0.92, 0.98, 0.96]
    },
    {
        "name": "Prosty chatbot FAQ",
        "vector": [0.20, 0.99, 0.50, 0.25, 0.10]
    },
    {
        "name": "AM Research Prototype",
        "vector": [0.92, 0.62, 0.78, 1.00, 0.99]
    },
]

feature_names = [
    "novelty",
    "feasibility",
    "business",
    "vision_alignment",
    "depth"
]

# wzorzec idei, której naprawdę szukamy
attractor = ResonantAttractor(
    target_vector=[0.90, 0.75, 0.85, 1.00, 0.95],
    feature_names=feature_names
)

best = attractor.select_best(ideas)
ranking = attractor.rank(ideas)

print("NAJLEPIEJ REZONUJĄCA IDEA:\n")
print(best["name"])

print("\nRANKING REZONANSU:\n")
for item in ranking:
    print(f'{item["name"]:30} -> {item["resonance"]:.4f}')
