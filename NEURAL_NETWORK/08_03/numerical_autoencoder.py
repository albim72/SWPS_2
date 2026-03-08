import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. GENEROWANIE PRZYKŁADOWYCH DANYCH NUMERYCZNYCH
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

n_normal = 4000
n_anomaly = 200
n_features = 20

# Dane normalne: kilka powiązanych zmiennych numerycznych
X_normal = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))

# Dodajemy zależności między cechami, żeby dane miały strukturę
X_normal[:, 5] = 0.7 * X_normal[:, 0] + 0.2 * X_normal[:, 1] + np.random.normal(0, 0.2, n_normal)
X_normal[:, 6] = -0.5 * X_normal[:, 2] + 0.3 * X_normal[:, 3] + np.random.normal(0, 0.2, n_normal)
X_normal[:, 7] = np.sin(X_normal[:, 4]) + np.random.normal(0, 0.1, n_normal)
X_normal[:, 8] = X_normal[:, 0] * X_normal[:, 2] + np.random.normal(0, 0.2, n_normal)

# Dane anomalne: inne rozkłady, przesunięcia, większa wariancja
X_anomaly = np.random.normal(loc=3.0, scale=2.2, size=(n_anomaly, n_features))
X_anomaly[:, 5] = -2.0 * X_anomaly[:, 0] + np.random.normal(0, 1.0, n_anomaly)
X_anomaly[:, 6] = 4.0 + np.random.normal(0, 1.5, n_anomaly)
X_anomaly[:, 7] = np.cos(X_anomaly[:, 4]) * 3 + np.random.normal(0, 0.5, n_anomaly)

# Etykiety tylko do ewaluacji
y_normal = np.zeros(n_normal, dtype=int)
y_anomaly = np.ones(n_anomaly, dtype=int)

# Łączymy zbiór testowy z normalnych i anomalii
X_all = np.vstack([X_normal, X_anomaly])
y_all = np.concatenate([y_normal, y_anomaly])


# ============================================================
# 2. PODZIAŁ DANYCH
# Trenujemy autoenkoder tylko na danych normalnych
# ============================================================

X_train_normal, X_val_normal = train_test_split(
    X_normal, test_size=0.2, random_state=42
)

X_test = X_all
y_test = y_all


# ============================================================
# 3. SKALOWANIE
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_val_scaled = scaler.transform(X_val_normal)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. BUDOWA MOCNEGO AUTOENKODERA
# ============================================================

input_dim = X_train_scaled.shape[1]
latent_dim = 4
l2_reg = 1e-4
drop_rate = 0.15

inputs = layers.Input(shape=(input_dim,), name="input_layer")

# ---------------- ENCODER ----------------
x = layers.Dense(
    128,
    kernel_regularizer=regularizers.l2(l2_reg),
    name="enc_dense_1"
)(inputs)
x = layers.BatchNormalization(name="enc_bn_1")(x)
x = layers.Activation("relu", name="enc_relu_1")(x)
x = layers.Dropout(drop_rate, name="enc_dropout_1")(x)

x = layers.Dense(
    64,
    kernel_regularizer=regularizers.l2(l2_reg),
    name="enc_dense_2"
)(x)
x = layers.BatchNormalization(name="enc_bn_2")(x)
x = layers.Activation("relu", name="enc_relu_2")(x)
x = layers.Dropout(drop_rate, name="enc_dropout_2")(x)

x = layers.Dense(
    32,
    kernel_regularizer=regularizers.l2(l2_reg),
    name="enc_dense_3"
)(x)
x = layers.BatchNormalization(name="enc_bn_3")(x)
x = layers.Activation("relu", name="enc_relu_3")(x)

latent = layers.Dense(latent_dim, activation=None, name="latent_space")(x)

# ---------------- DECODER ----------------
x = layers.Dense(
    32,
    kernel_regularizer=regularizers.l2(l2_reg),
    name="dec_dense_1"
)(latent)
x = layers.BatchNormalization(name="dec_bn_1")(x)
x = layers.Activation("relu", name="dec_relu_1")(x)

x = layers.Dense(
    64,
    kernel_regularizer=regularizers.l2(l2_reg),
    name="dec_dense_2"
)(x)
x = layers.BatchNormalization(name="dec_bn_2")(x)
x = layers.Activation("relu", name="dec_relu_2")(x)
x = layers.Dropout(drop_rate, name="dec_dropout_2")(x)

x = layers.Dense(
    128,
    kernel_regularizer=regularizers.l2(l2_reg),
    name="dec_dense_3"
)(x)
x = layers.BatchNormalization(name="dec_bn_3")(x)
x = layers.Activation("relu", name="dec_relu_3")(x)

outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

autoencoder = Model(inputs, outputs, name="strong_numeric_autoencoder")

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

autoencoder.summary()


# ============================================================
# 5. CALLBACKI
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# 6. TRENING
# ============================================================

history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    validation_data=(X_val_scaled, X_val_scaled),
    epochs=200,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# ============================================================
# 7. REKONSTRUKCJA I BŁĄD
# ============================================================

X_train_recon = autoencoder.predict(X_train_scaled, verbose=0)
X_val_recon = autoencoder.predict(X_val_scaled, verbose=0)
X_test_recon = autoencoder.predict(X_test_scaled, verbose=0)

train_mse = np.mean(np.square(X_train_scaled - X_train_recon), axis=1)
val_mse = np.mean(np.square(X_val_scaled - X_val_recon), axis=1)
test_mse = np.mean(np.square(X_test_scaled - X_test_recon), axis=1)

print("\nŚredni błąd rekonstrukcji TRAIN:", np.mean(train_mse))
print("Średni błąd rekonstrukcji VAL:", np.mean(val_mse))
print("Średni błąd rekonstrukcji TEST:", np.mean(test_mse))


# ============================================================
# 8. PRÓG ANOMALII
# Ustalamy go na podstawie walidacyjnych danych normalnych
# ============================================================

threshold = np.percentile(val_mse, 95)
print("\nPróg anomalii:", threshold)

y_pred = (test_mse > threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


# ============================================================
# 9. WYKRESY
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Historia treningu autoenkodera")
plt.xlabel("Epoka")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(test_mse[y_test == 0], bins=50, alpha=0.7, label="Normal")
plt.hist(test_mse[y_test == 1], bins=50, alpha=0.7, label="Anomaly")
plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")
plt.title("Rozkład błędu rekonstrukcji")
plt.xlabel("Reconstruction MSE")
plt.ylabel("Liczba próbek")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 10. OSOBNY ENCODER
# Żeby wyciągać reprezentacje latentne
# ============================================================

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("latent_space").output)

latent_train = encoder.predict(X_train_scaled, verbose=0)
latent_test = encoder.predict(X_test_scaled, verbose=0)

latent_df = pd.DataFrame(latent_test, columns=[f"z{i+1}" for i in range(latent_dim)])
latent_df["label"] = y_test
latent_df["reconstruction_error"] = test_mse

print("\nPrzykładowe reprezentacje latentne:")
print(latent_df.head())
