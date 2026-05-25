# ==============================
# PRACTICAL NO. 5
# AIM: Neural Networks with TensorFlow/Keras
# Dataset: MNIST (Handwritten Digits)
# ==============================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Load & Preprocess MNIST ──────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32")  / 255

# Flatten for Dense models
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat  = x_test.reshape(-1, 28*28)

print("Train shape:", x_train.shape, y_train.shape)

# ── 2. Architecture 1: Simple Dense Model ───────────────────────────────────
def build_dense_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64,  activation='relu'),
        layers.Dense(10,  activation='softmax')
    ])
    return model

model = build_dense_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    x_train_flat, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
print("\nDense Model Evaluation:")
model.evaluate(x_test_flat, y_test)

# ── 3. Optimizer Comparison ─────────────────────────────────────────────────
print("\n--- Optimizer Comparison ---")
for opt in ['adam', 'sgd', 'rmsprop']:
    print(f"\nTraining with optimizer: {opt}")
    m = build_dense_model()
    m.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    m.fit(x_train_flat, y_train, epochs=3, batch_size=32, verbose=0)
    loss, acc = m.evaluate(x_test_flat, y_test, verbose=0)
    print(f"{opt} accuracy: {acc:.4f}")

# ── 4. Architecture 2: Deep Dense Model ─────────────────────────────────────
def build_deep_model():
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64,  activation='relu'),
        layers.Dense(32,  activation='relu'),
        layers.Dense(10,  activation='softmax')
    ])
    return model

# ── 5. Architecture 3: Dense with Dropout ───────────────────────────────────
def build_dropout_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(64,  activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10,  activation='softmax')
    ])
    return model

# ── 6. Architecture 4: CNN ──────────────────────────────────────────────────
x_train_cnn = x_train[..., None]
x_test_cnn  = x_test[..., None]

def build_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model_cnn = build_cnn()
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_cnn.fit(x_train_cnn, y_train, epochs=5, validation_split=0.1)
print("\nCNN Evaluation:")
model_cnn.evaluate(x_test_cnn, y_test)

# ── 7. Optimizer x Depth Grid ───────────────────────────────────────────────
print("\n--- Optimizer x Depth Results ---")
results = []
for opt in ['adam', 'sgd']:
    for layers_num in [2, 4]:
        m = keras.Sequential()
        m.add(layers.Dense(128, activation='relu', input_shape=(784,)))
        for _ in range(layers_num - 1):
            m.add(layers.Dense(64, activation='relu'))
        m.add(layers.Dense(10, activation='softmax'))
        m.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        m.fit(x_train_flat, y_train, epochs=3, verbose=0)
        _, acc = m.evaluate(x_test_flat, y_test, verbose=0)
        results.append((opt, layers_num, round(acc, 4)))

print(results)
