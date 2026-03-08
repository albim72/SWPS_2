# ==========================================
# SEKCJA 3
# Dwa atraktory
# ==========================================

# -------------------------------------------------
# Atraktor 1: ambitny / mocny / wysoko jakościowy
# Atraktor 2: umiarkowany / stabilny / bardziej zachowawczy
# -------------------------------------------------

attractor_1 = np.array([0.90, 0.85, 0.80, 0.80], dtype=np.float32)
attractor_2 = np.array([0.72, 0.68, 0.62, 0.65], dtype=np.float32)

# odległości
distance_A1 = np.linalg.norm(X - attractor_1, axis=1)
distance_A2 = np.linalg.norm(X - attractor_2, axis=1)

# rezonanse
resonance_A1 = 1 / (1 + distance_A1)
resonance_A2 = 1 / (1 + distance_A2)

# minimalna odległość do najbliższego atraktora
best_distance = np.minimum(distance_A1, distance_A2)

# maksymalny rezonans z dwóch atraktorów
best_resonance = np.maximum(resonance_A1, resonance_A2)

results_attr2 = df.copy()
results_attr2["distance_A1"] = distance_A1
results_attr2["distance_A2"] = distance_A2
results_attr2["resonance_A1"] = resonance_A1
results_attr2["resonance_A2"] = resonance_A2
results_attr2["best_resonance"] = best_resonance

print("Wyniki względem dwóch atraktorów:")
print(results_attr2.sort_values("best_resonance", ascending=False))
