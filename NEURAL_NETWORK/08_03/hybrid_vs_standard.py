import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models

np.random.seed(42)

# 1. Generujemy dane
X = np.random.rand(500, 4).astype(np.float32)
attractor = np.array([0.9, 0.8, 0.2, 0.2], dtype=np.float32)

# etykieta zależna od odległości
distances = np.linalg.norm(X - attractor, axis=1)
y = (distances < 0.45).astype(np.float32)

# 2. Wersja zwykła i hybrydowa
X_plain = X
X_hybrid = np.hstack([X, distances.reshape(-1, 1)])

# 3. Ten sam podział train/test
X_train_plain, X_test_plain, y_train, y_test = train_test_split(
    X_plain, y, test_size=0.3, random_state=42
)

X_train_hybrid, X_test_hybrid, _, _ = train_test_split(
    X_hybrid, y, test_size=0.3, random_state=42
)

# 4. Funkcja budująca model
def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(16, activation="relu", input_shape=(input_dim,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# 5. Model zwykły
model_plain = build_model(4)
model_plain.fit(X_train_plain, y_train, epochs=30, verbose=0)
pred_plain = (model_plain.predict(X_test_plain) > 0.5).astype(int)

# 6. Model hybrydowy
model_hybrid = build_model(5)
model_hybrid.fit(X_train_hybrid, y_train, epochs=30, verbose=0)
pred_hybrid = (model_hybrid.predict(X_test_hybrid) > 0.5).astype(int)

# 7. Porównanie
acc_plain = accuracy_score(y_test, pred_plain)
acc_hybrid = accuracy_score(y_test, pred_hybrid)

print("Accuracy plain:", acc_plain)
print("Accuracy hybrid:", acc_hybrid)
