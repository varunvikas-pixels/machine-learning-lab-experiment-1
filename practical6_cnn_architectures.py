# ==============================
# PRACTICAL NO. 6
# AIM: CNN Architectures — Simple, Deep & Transfer Learning
# Dataset: MNIST (Handwritten Digits)
# ==============================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_score
import numpy as np

# ── 1. Load & Preprocess ────────────────────────────────────────────────────
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)) / 255.0
test_images  = test_images.reshape((test_images.shape[0],  28, 28, 1)) / 255.0

# ── 2. Model 1: Simple CNN ──────────────────────────────────────────────────
def build_simple_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,  activation='relu'),
        layers.Dense(10,  activation='softmax')
    ])
    return model

# ── 3. Model 2: Deep CNN with Dropout ───────────────────────────────────────
def build_deep_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10,  activation='softmax')
    ])
    return model

# ── 4. Model 3: Transfer Learning (MobileNetV2) ─────────────────────────────
def build_transfer_model():
    input_tensor = keras.Input(shape=(28, 28, 1))
    x = layers.Lambda(lambda img: tf.image.resize(img, (32, 32)))(input_tensor)
    x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
    base_model = keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None
    )
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs=input_tensor, outputs=output_tensor)

# ── 5. Compile & Train ──────────────────────────────────────────────────────
def compile_and_train(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_images, train_labels,
        epochs=10,
        validation_split=0.2,
        batch_size=64
    )
    return history

# ── 6. Evaluate (Accuracy + Precision) ──────────────────────────────────────
def evaluate_model(model):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    y_pred         = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision      = precision_score(test_labels, y_pred_classes, average='macro')
    return test_acc, precision

# ── 7. Run All Experiments ──────────────────────────────────────────────────
models = {
    "Simple CNN"    : build_simple_cnn(),
    "Deep CNN"      : build_deep_cnn(),
    "Transfer Model": build_transfer_model()
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    compile_and_train(model)
    acc, prec = evaluate_model(model)
    results[name] = {"Accuracy": round(acc, 4), "Precision": round(prec, 4)}

print("\n--- Final Results ---")
for name, metrics in results.items():
    print(f"{name}: {metrics}")
