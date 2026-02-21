"""
train_model.py — Training pipeline for the Hand Gesture Classifier.

Overview
--------
Reads hand landmark data (42 features: 21 landmarks × x,y) from a CSV,
normalises features with StandardScaler, trains a regularised Dense neural
network with early stopping, evaluates performance, and persists the artefacts
needed for real-time inference:

  * model.h5          — Keras saved model
  * scaler_mean.npy   — StandardScaler mean vector
  * scaler_scale.npy  — StandardScaler scale vector

Usage
-----
    python train_model.py
    python train_model.py --epochs 100 --lr 0.0005 --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_model")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Hand Gesture Classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=cfg.DATASET_DIR / "gesture_landmarks.csv",
                   help="Path to the landmark CSV file.")
    p.add_argument("--epochs",     type=int,   default=cfg.training.epochs)
    p.add_argument("--batch",      type=int,   default=cfg.training.batch_size)
    p.add_argument("--lr",         type=float, default=cfg.training.learning_rate)
    p.add_argument("--test-size",  type=float, default=cfg.training.test_size)
    p.add_argument("--no-early-stop", action="store_true",
                   help="Disable early stopping.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load and preprocess the landmark CSV.

    Returns:
        X (np.ndarray): Feature matrix, shape (N, 42).
        y (np.ndarray): Integer-encoded label vector, shape (N,).
        encoder (LabelEncoder): Fitted encoder (for class-name lookup).
    """
    logger.info("Loading dataset: %s", csv_path)
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    label_col = "label" if "label" in data.columns else "gesture_label"
    if label_col not in data.columns:
        raise ValueError(
            f"Expected a column named 'label' or 'gesture_label'. "
            f"Found: {list(data.columns)}"
        )

    X = data.drop(columns=[label_col]).values.astype(np.float32)
    raw_y = data[label_col].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y)

    logger.info(
        "Dataset loaded — samples: %d, features: %d, classes: %s",
        X.shape[0], X.shape[1], list(encoder.classes_),
    )
    return X, y, encoder


def normalise(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler and return the normalised feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    np.save(cfg.SCALER_MEAN,  scaler.mean_)
    np.save(cfg.SCALER_SCALE, scaler.scale_)
    logger.info("Scaler saved → %s, %s", cfg.SCALER_MEAN, cfg.SCALER_SCALE)
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_dim: int, num_classes: int, lr: float) -> tf.keras.Model:
    """Construct and compile the classifier network.

    Architecture
    ------------
    Input(42) → Dense(128, ReLU) → Dropout(0.3)
              → Dense(64,  ReLU) → Dropout(0.3)
              → Dense(num_classes, Softmax)
    """
    tc = cfg.training
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                tc.dense1_units, activation="relu",
                input_shape=(input_dim,), name="hidden_1",
            ),
            tf.keras.layers.Dropout(tc.dropout_rate, name="drop_1"),
            tf.keras.layers.Dense(tc.dense2_units, activation="relu", name="hidden_2"),
            tf.keras.layers.Dropout(tc.dropout_rate, name="drop_2"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="gesture_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


def get_callbacks(patience: int, disable: bool) -> list:
    """Return Keras training callbacks."""
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
    ]
    if not disable:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    history: tf.keras.callbacks.History,
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val_cat: np.ndarray,
) -> None:
    """Log final metrics and overfitting gap."""
    train_acc = history.history["accuracy"][-1] * 100
    val_acc   = history.history["val_accuracy"][-1] * 100
    gap       = abs(train_acc - val_acc)

    _, best_val = model.evaluate(X_val, y_val_cat, verbose=0)
    best_val   *= 100

    logger.info("─" * 50)
    logger.info("Training Accuracy   : %.2f %%", train_acc)
    logger.info("Validation Accuracy : %.2f %%", val_acc)
    logger.info("Best Val Accuracy   : %.2f %%", best_val)
    logger.info("Overfitting Gap     : %.2f %%", gap)
    if gap > 10:
        logger.warning("⚠  Gap > 10%% — consider more data or stronger Dropout.")
    else:
        logger.info("✅ Model generalises well (gap ≤ 10%%).")
    logger.info("─" * 50)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 1. Load data
    X, y, encoder = load_dataset(args.csv)
    num_classes   = len(encoder.classes_)

    # 2. Normalise
    X_scaled, _ = normalise(X)

    # 3. Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=args.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)

    # 4. Build & train
    model = build_model(X.shape[1], num_classes, args.lr)
    callbacks = get_callbacks(
        cfg.training.early_stopping_patience, args.no_early_stop
    )

    logger.info("Training started (epochs=%d, batch=%d, lr=%s) …",
                args.epochs, args.batch, args.lr)
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluate
    evaluate(history, model, X_val, y_val_cat)

    # 6. Persist
    model.save(cfg.MODEL_PATH)
    logger.info("✅ Model saved → %s", cfg.MODEL_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
