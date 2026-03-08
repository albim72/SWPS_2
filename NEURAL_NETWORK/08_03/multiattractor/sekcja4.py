# ==========================================
# SEKCJA 4
# Hybryda: sieć + cechy atraktorowe
# ==========================================

# -------------------------------------------------
# 1. Budujemy nowe cechy atraktorowe
# -------------------------------------------------

distance_A1 = np.linalg.norm(X - attractor_1, axis=1).reshape(-1, 1)
distance_A2 = np.linalg.norm(X - attractor_2, axis=1).reshape(-1, 1)

resonance_A1 = (1 / (1 + distance_A1))
resonance_A2 = (1 / (1 + distance_A2))

# -------------------------------------------------
# 2. Łączymy surowe cechy z cechami atraktorowymi
# Oryginalnie mieliśmy 4 cechy
# Teraz dodajemy 4 kolejne:
# distance_A1, distance_A2, resonance_A1, resonance_A2
# -------------------------------------------------

X_hybrid = np.hstack([
    X,
    distance_A1,
    distance_A2,
    resonance_A1,
    resonance_A2
]).astype(np.float32)

hybrid_feature_names = [
    "novelty", "feasibility", "cost_efficiency", "stability",
    "distance_A1", "distance_A2", "resonance_A1", "resonance_A2"
]

df_hybrid = pd.DataFrame(X_hybrid, columns=hybrid_feature_names)
df_hybrid["label"] = y.astype(int)

print("Dane hybrydowe:")
print(df_hybrid)

# -------------------------------------------------
# 3. Budujemy nową sieć dla 8 cech wejściowych
# -------------------------------------------------

model_hybrid = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(12, activation="relu"),
    layers.Dense(6, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model_hybrid.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------
# 4. Uczymy model hybrydowy
# -------------------------------------------------

history_hybrid = model_hybrid.fit(
    X_hybrid, y,
    epochs=120,
    verbose=0
)

# -------------------------------------------------
# 5. Predykcje modelu hybrydowego
# -------------------------------------------------

hybrid_pred = model_hybrid.predict(X_hybrid, verbose=0).flatten()
hybrid_class = (hybrid_pred > 0.5).astype(int)

results_final = df.copy()
results_final["base_nn_score"] = base_pred
results_final["base_nn_class"] = base_class
results_final["hybrid_nn_score"] = hybrid_pred
results_final["hybrid_nn_class"] = hybrid_class
results_final["distance_A1"] = distance_A1.flatten()
results_final["distance_A2"] = distance_A2.flatten()
results_final["resonance_A1"] = resonance_A1.flatten()
results_final["resonance_A2"] = resonance_A2.flatten()

print("\nPorównanie modelu bazowego i hybrydowego:")
print(results_final)
