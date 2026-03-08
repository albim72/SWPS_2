import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# -----------------------------
# 1. Dane
# -----------------------------
X = np.array([
    [0.9, 0.8, 0.3, 0.2],
    [0.7, 0.6, 0.5, 0.4],
    [0.4, 0.5, 0.8, 0.7],
    [0.85, 0.75, 0.2, 0.3],
    [0.3, 0.4, 0.9, 0.8]
], dtype=np.float32)

y = np.array([1, 1, 0, 1, 0], dtype=np.float32).reshape(-1, 1)

# -----------------------------
# 2. Funkcja budująca model
# -----------------------------
def build_attractor_model(input_dim=4, hidden_dim=8, embedding_dim=2):
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    
    hidden = layers.Dense(hidden_dim, activation="relu", name="hidden_dense")(inputs)
    
    embedding = layers.Dense(
        embedding_dim,
        activation=None,
        name="embedding"
    )(hidden)
    
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="classifier_output"
    )(embedding)

    model = Model(
        inputs=inputs,
        outputs=[outputs, embedding],
        name="AttractorEmbeddingModel"
    )
    return model

# -----------------------------
# 3. Utworzenie modelu
# -----------------------------
model = build_attractor_model()

# pokaż architekturę
model.summary()

# -----------------------------
# 4. Atraktor w przestrzeni embeddingu
# -----------------------------
embedding_attractor = tf.constant([[1.0, 1.0]], dtype=tf.float32)

# -----------------------------
# 5. Optymalizator
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# -----------------------------
# 6. Pętla treningowa
# -----------------------------
for epoch in range(100):
    with tf.GradientTape() as tape:
        preds, emb = model(X, training=True)

        classification_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y, preds)
        )

        attractor_loss = tf.reduce_mean(
            tf.reduce_sum((emb - embedding_attractor) ** 2, axis=1)
        )

        total_loss = classification_loss + 0.1 * attractor_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 20 == 0:
        print(
            f"epoch={epoch}, "
            f"class_loss={classification_loss.numpy():.4f}, "
            f"attr_loss={attractor_loss.numpy():.4f}, "
            f"total={total_loss.numpy():.4f}"
        )

# -----------------------------
# 7. Predykcja końcowa
# -----------------------------
preds, emb = model(X, training=False)

print("\nPredictions:")
print(preds.numpy())

print("\nEmbeddings:")
print(emb.numpy())
