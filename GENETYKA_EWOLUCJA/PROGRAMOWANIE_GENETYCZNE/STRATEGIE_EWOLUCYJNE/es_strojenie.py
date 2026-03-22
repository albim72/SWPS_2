import random

# Parametry agenta:
# [odwaga, ostrożność, eksploracja]
# zakres umowny: 0..10

def simulate_agent(params):
    courage, caution, exploration = params

    # Przykładowy fitness:
    # chcemy balans, nie skrajności
    score = 0
    score += 12 - abs(courage - 7)
    score += 12 - abs(caution - 5)
    score += 12 - abs(exploration - 8)

    # kara za nadmierne skrajności
    if courage > 9 and caution < 2:
        score -= 5

    return score


def mutate(params, sigma=1.0):
    child = [p + random.gauss(0, sigma) for p in params]
    # przycinanie do zakresu 0..10
    child = [max(0, min(10, x)) for x in child]
    return child


parent = [random.uniform(0, 10) for _ in range(3)]

for gen in range(20):
    child = mutate(parent, sigma=0.9)

    if simulate_agent(child) > simulate_agent(parent):
        parent = child

    print(
        f"Gen {gen:02d} | "
        f"params={[round(x, 2) for x in parent]} | "
        f"fitness={simulate_agent(parent):.3f}"
    )

print("\nNajlepszy agent:", [round(x, 2) for x in parent])
print("Fitness:", simulate_agent(parent))
