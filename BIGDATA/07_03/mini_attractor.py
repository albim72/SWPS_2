import math

data = [
    {"name":"A","score":72},
    {"name":"B","score":91},
    {"name":"C","score":87},
    {"name":"D","score":95},
    {"name":"E","score":60},
]

attractor = 90

best = min(data, key=lambda x: abs(x["score"] - attractor))
print(best["name"])
print(best["score"])
print("_"*70)
best_3 = sorted(data, key=lambda x: abs(x["score"] - attractor))[:3]
for item in best_3:
    print(item["name"], item["score"])

print("_"*70)

data = [
    {"name":"A","features":[0.2,0.8,0.5]},
    {"name":"B","features":[0.9,0.1,0.3]},
    {"name":"C","features":[0.4,0.7,0.6]},
    {"name":"D","features":[0.1,0.7,0.4]},
    {"name":"E","features":[0.2,0.4,0.5]},
]

attractor = [0.3,0.8,0.5]

def distance(v1,v2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(v1,v2)))

best = min(data, key=lambda x: distance(x["features"],attractor))
print(best["name"])
print("_"*70)
print(f"odległość od atraktora: {distance(attractor,best['features'])}")
