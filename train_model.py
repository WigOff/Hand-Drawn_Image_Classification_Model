"""
train_model.py — Quick Draw CNN Classifier
==========================================
Dataset : "Google QuickDraw" – 10 selected categories
Source  : https://www.kaggle.com/datasets/google-brain/quickdraw-dataset
          (or the smaller pre-packaged version at
           https://www.kaggle.com/datasets/ashishjangra27/quickdraw-dataset-10-categories)

HOW TO USE
----------
1.  Install the Kaggle CLI and place kaggle.json in ~/.kaggle/
2.  Run:
        python train_model.py

The script will:
  • Download the dataset (≈ 180 MB for 10 categories)
  • Train a CNN (expects ≥ 90 % val-accuracy)
  • Save  model/quickdraw_model.keras
  • Save  model/class_names.json
"""

import os, json, zipfile, shutil, urllib.request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CLASSES = [
    "airplane", "bicycle", "bird", "cat",
    "dog",      "fish",    "flower", "house",
    "tree",     "umbrella"
]
SAMPLES_PER_CLASS = 10_000   # 10 k × 10 classes = 100 k total  (fast, still ≥90 %)
IMG_SIZE          = 28
BATCH_SIZE        = 256
EPOCHS            = 25
MODEL_DIR         = "model"
DATA_DIR          = "quickdraw_data"

# ──────────────────────────────────────────────
# DOWNLOAD  (two strategies)
# ──────────────────────────────────────────────

def download_with_kaggle_api():
    """Requires ~/.kaggle/kaggle.json"""
    import subprocess
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        npy = os.path.join(DATA_DIR, f"{cls}.npy")
        if os.path.exists(npy):
            print(f"  ✓ {cls}.npy already present, skipping download")
            continue
        print(f"  ↓ Downloading {cls} …")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "ashishjangra27/quickdraw-dataset-10-categories",
            "-f", f"{cls}.npy",
            "-p", DATA_DIR,
            "--unzip"
        ], check=True)


def download_directly():
    """
    Falls back to Google's public npy files hosted on storage.googleapis.com.
    No Kaggle account needed.
    """
    BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        dest = os.path.join(DATA_DIR, f"{cls}.npy")
        if os.path.exists(dest):
            print(f"  ✓ {cls}.npy already present")
            continue
        url = f"{BASE}/{cls.replace(' ', '%20')}.npy"
        print(f"  ↓ {url}")
        urllib.request.urlretrieve(url, dest)
    print("Download complete.\n")


def ensure_data():
    npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")] if os.path.isdir(DATA_DIR) else []
    if len(npy_files) >= len(CLASSES):
        print("All .npy files found — skipping download.\n")
        return

    # try kaggle API first, fall back to direct
    try:
        download_with_kaggle_api()
    except Exception as e:
        print(f"Kaggle API unavailable ({e}). Falling back to direct Google download …")
        download_directly()


# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────

def load_dataset():
    print("Loading and preprocessing data …")
    X, y = [], []
    for label_idx, cls in enumerate(CLASSES):
        path = os.path.join(DATA_DIR, f"{cls}.npy")
        data = np.load(path)                            # shape (N, 784)
        data = data[:SAMPLES_PER_CLASS]
        X.append(data)
        y.extend([label_idx] * len(data))
        print(f"  {cls:12s}: {len(data)} samples loaded")

    X = np.concatenate(X, axis=0).astype("float32") / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)           # (N,28,28,1)
    y = np.array(y, dtype="int32")

    # shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  |  Val: {len(X_val)}\n")
    return X_train, X_val, y_train, y_val


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ──────────────────────────────────────────────
# DATA AUGMENTATION
# ──────────────────────────────────────────────

def make_dataset(X, y, augment=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        def aug(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.keras.layers.RandomRotation(0.08)(img, training=True)
            img = tf.keras.layers.RandomZoom(0.10)(img, training=True)
            return img, label
        ds = ds.shuffle(10_000).map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train():
    ensure_data()
    X_train, X_val, y_train, y_val = load_dataset()

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = build_model(len(CLASSES))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        ),
    ]

    train_ds = make_dataset(X_train, y_train, augment=True)
    val_ds   = make_dataset(X_val,   y_val,   augment=False)

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks
    )

    # ── save final model + class map ──
    final_path = os.path.join(MODEL_DIR, "quickdraw_model.keras")
    model.save(final_path)
    print(f"\n✅ Model saved → {final_path}")

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(CLASSES, f, indent=2)
    print("✅ Class names saved → model/class_names.json")

    # ── evaluation ──
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n📊 Final val-accuracy : {val_acc * 100:.2f} %")
    if val_acc >= 0.90:
        print("🎉 Meets the ≥ 90 % accuracy target!\n")
    else:
        print("⚠  Below 90 % — try increasing EPOCHS or SAMPLES_PER_CLASS.\n")

    # ── learning-curve plot ──
    _plot_history(history)
    return model


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(path)
    print(f"📈 Training curves saved → {path}")
    plt.close()


if __name__ == "__main__":
    # reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    train()
