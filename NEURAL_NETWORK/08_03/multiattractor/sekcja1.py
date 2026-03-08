# ==========================================
# SEKCJA 1
# Zwykła sieć neuronowa na prostych danych
# ==========================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# dla powtarzalności wyników
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------
# 1. Tworzymy mały zbiór danych
# Każdy obiekt ma 4 cechy:
# novelty, feasibility, cost_efficiency, stability
# y = 1 oznacza "dobry projekt"
# y = 0 oznacza "słaby projekt"
# -------------------------------------------------

X = np.array([
    [0.90, 0.85, 0.80, 0.75],
    [0.88, 0.80, 0.78, 0.70],
    [0.20, 0.30, 0.40, 0.35],
    [0.25, 0.20, 0.30, 0.25],
    [0.75, 0.70, 0.65, 0.60],
    [0.35, 0.40, 0.45, 0.30],
    [0.92, 0.90, 0.85, 0.88],
    [0.15, 0.25, 0.20, 0.22],
    [0.78, 0.82, 0.74, 0.76],
    [0.28, 0.35, 0.38, 0.32],
    [0.83, 0.79, 0.81, 0.72],
    [0.18, 0.22, 0.33, 0.27],
], dtype=np.float32)

y = np.array([
    1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0
], dtype=np.float32)

feature_names = ["novelty", "feasibility", "cost_efficiency", "stability"]

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y.astype(int)

print("Dane wejściowe:")
print(df)

# -------------------------------------------------
# 2. Budujemy prostą sieć neuronową
# -------------------------------------------------

model_base = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model_base.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------
# 3. Uczymy model
# -------------------------------------------------

history_base = model_base.fit(
    X, y,
    epochs=100,
    verbose=0
)

# -------------------------------------------------
# 4. Predykcje
# -------------------------------------------------

base_pred = model_base.predict(X, verbose=0).flatten()
base_class = (base_pred > 0.5).astype(int)

results_base = df.copy()
results_base["nn_score"] = base_pred
results_base["nn_class"] = base_class

print("\nWyniki zwykłej sieci:")
print(results_base)
