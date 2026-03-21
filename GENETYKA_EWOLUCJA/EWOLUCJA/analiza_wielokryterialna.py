#architektura decyzji: wybór najalepszego kandydata  projektu

#szukamy najlepszego kandydata do projektu AI
"""

Architektura decyzji

1. obiekty decyzji - co wybieramy?
2. kryteria - na jakiej podstawie?
3. wagi - co jest ważniejsze?
4. reguły oceny - ja połączyć kryteria w jedną ocenę?
5. wynik i interpretacja
"""
#dane do przykałdu

# candidates = [
#     {"name":"Anna","tech":9,"communication":6,"cost":7,"availability":8,"fit":9},
#     {"name":"Bartek","tech":7,"communication":9,"cost":5,"availability":9,"fit":8},
#     {"name":"Celina","tech":8,"communication":7,"cost":6,"availability":6,"fit":10},
#     {"name":"Dawid","tech":6,"communication":8,"cost":3,"availability":10,"fit":7},
# ]
"""
prompt do gpt:
candidates = [
    {"name":"Anna","tech":9,"communication":6,"cost":7,"availability":8,"fit":9},
    {"name":"Bartek","tech":7,"communication":9,"cost":5,"availability":9,"fit":8},
    {"name":"Celina","tech":8,"communication":7,"cost":6,"availability":6,"fit":10},
    {"name":"Dawid","tech":6,"communication":8,"cost":3,"availability":10,"fit":7},
]

zbudyj źródło dla różnych imion zarówno żeńskicjh jak i męskich przyszłych pracowników projwktu  ocena każdego prametryu mieścci się w zakresie 0-10, zbuduj listę 1000 słowników
"""
import random

# Lista imion żeńskich
female_names = [
    "Anna", "Maria", "Katarzyna", "Agnieszka", "Ewa", "Magdalena", "Joanna",
    "Aleksandra", "Monika", "Karolina", "Paulina", "Natalia", "Marta",
    "Barbara", "Dorota", "Elżbieta", "Zofia", "Patrycja", "Weronika", "Emilia",
    "Julia", "Oliwia", "Wiktoria", "Helena", "Alicja"
]

# Lista imion męskich
male_names = [
    "Bartek", "Piotr", "Michał", "Tomasz", "Krzysztof", "Paweł", "Marcin",
    "Adam", "Jakub", "Mateusz", "Dawid", "Łukasz", "Kamil", "Grzegorz",
    "Rafał", "Sebastian", "Szymon", "Maciej", "Jan", "Wojciech",
    "Norbert", "Karol", "Patryk", "Damian", "Filip"
]

# Połączona lista imion
all_names = female_names + male_names

# Parametry kandydatów
parameters = ["tech", "communication", "cost", "availability", "fit"]

# Generator 1000 kandydatów
candidates = []

for i in range(1000):
    candidate = {
        "id": i + 1,
        "name": random.choice(all_names),
        "tech": random.randint(0, 10),
        "communication": random.randint(0, 10),
        "cost": random.randint(0, 10),
        "availability": random.randint(0, 10),
        "fit": random.randint(0, 10)
    }
    candidates.append(candidate)

# Podgląd pierwszych 10 rekordów
for c in candidates[:10]:
    print(c)

print(f"\nLiczba wygenerowanych kandydatów: {len(candidates)}")


#wagi kryteriów
weights = {
    "tech": 0.30,
    "communication": 0.20,
    "cost": 0.15,
    "availability": 0.15,
    "fit": 0.20
}
#funkcja licząca wynik końcowy dla jednego kandydata
def calculate_score(candidate,weights):
    #koszt działa odwrotnie - im mniej tym lepiej
    adjusted_cost = 10 - candidate["cost"]

    score = (
        candidate["tech"] * weights["tech"] +
        candidate["communication"] * weights["communication"] +
        adjusted_cost * weights["cost"] +
        candidate["availability"] * weights["availability"] +
        candidate["fit"] * weights["fit"]
    )

    return round(score, 2)

#oblicz wynik dla każego kandyata
for candidate in candidates:
    candidate["score"] = calculate_score(candidate,weights)

#sortowanie od najlepszego do najsłabszego
candidates.sort(key=lambda x: x["score"], reverse=True)

print("\nNajlepsi kandydaci:")
for c in candidates[:10]:
    print(f"{c['name']} - {c['score']}")

#najlepszy kandydat
print(f"\nNajlepszy kandydat: {candidates[0]['name']} - {candidates[0]['score']}")
#szczegóły:
best = candidates[0]
print("Szczegóły")
print(f"Technologie: {best['tech']}")
print(f"Comunikacja: {best['communication']}")
print(f"Cost: {best['cost']}")
print(f"Dostępnosc: {best['availability']}")
print(f"Fit: {best['fit']}")
