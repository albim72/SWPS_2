# ==========================================
# SEKCJA 2
# Jeden atraktor jako wzorzec idealny
# ==========================================

# -------------------------------------------------
# 1. Definiujemy jeden atraktor
# To jest ręcznie ustalony punkt idealny
# -------------------------------------------------

attractor_1 = np.array([0.90, 0.85, 0.80, 0.80], dtype=np.float32)

print("Atraktor 1:")
print(attractor_1)

# -------------------------------------------------
# 2. Liczymy odległość euklidesową od atraktora
# -------------------------------------------------

distance_1 = np.linalg.norm(X - attractor_1, axis=1)

# -------------------------------------------------
# 3. Zamieniamy odległość na rezonans
# Im bliżej, tym większy rezonans
# -------------------------------------------------

resonance_1 = 1 / (1 + distance_1)

results_attr1 = df.copy()
results_attr1["distance_to_A1"] = distance_1
results_attr1["resonance_A1"] = resonance_1

print("\nWyniki względem jednego atraktora:")
print(results_attr1.sort_values("resonance_A1", ascending=False))
