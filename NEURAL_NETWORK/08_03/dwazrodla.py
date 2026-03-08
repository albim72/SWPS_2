import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

X = np.array([
    [0.9, 0.8, 0.3, 0.2],
    [0.7, 0.6, 0.5, 0.4],
    [0.4, 0.5, 0.8, 0.7],
    [0.85, 0.75, 0.2, 0.3],
    [0.3, 0.4, 0.9, 0.8]
], dtype=np.float32)

y = np.array([1, 1, 0, 1, 0], dtype=np.float32)

attractor = np.array([0.9, 0.8, 0.2, 0.2], dtype=np.float32)

model = models.Sequential([
    layers.Dense(16, activation="relu", input_shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=50, verbose=0)

nn_pred = model.predict(X).flatten()

distances = np.linalg.norm(X - attractor, axis=1)
resonance = 1 / (1 + distances)

final_score = 0.7 * nn_pred + 0.3 * resonance
final_class = (final_score > 0.5).astype(int)

print("NN prediction:", nn_pred)
print("Resonance:", resonance)
print("Final score:", final_score)
print("Final class:", final_class)
