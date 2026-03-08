import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

X = np.array([
    [0.9, 0.8, 0.3, 0.2],
    [0.7, 0.6, 0.5, 0.4],
    [0.4, 0.5, 0.8, 0.7],
    [0.85, 0.75, 0.2, 0.3],
    [0.3, 0.4, 0.9, 0.8]
], dtype=np.float32)

y = np.array([1, 1, 0, 1, 0], dtype=np.float32).reshape(-1, 1)

inputs = layers.Input(shape=(4,))
hidden = layers.Dense(8, activation="relu")(inputs)
embedding = layers.Dense(2, activation=None, name="embedding")(hidden)
outputs = layers.Dense(1, activation="sigmoid")(embedding)

model = Model(inputs=inputs, outputs=[outputs, embedding])

# atraktor w przestrzeni embeddingu
embedding_attractor = tf.constant([[1.0, 1.0]], dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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

preds, emb = model(X, training=False)
print("Predictions:\n", preds.numpy())
print("Embeddings:\n", emb.numpy())
