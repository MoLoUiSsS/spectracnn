"""Numpy metrics and optional matplotlib plots (no runtime ML dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, prediction in zip(np.asarray(y_true), np.asarray(y_pred), strict=True):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []

    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
            "f1": round(float(f1), 5),
            "support": support,
        }
        f1_values.append(f1)

    return {
        "accuracy": round(accuracy, 5),
        "macro_f1": round(float(np.mean(f1_values)), 5),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "n_test": total,
    }


def save_training_curves(history: dict[str, list[float]], path: str | Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_per_snr_curve(rows: list[dict[str, float]], path: str | Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snr = [row["snr_db"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(snr, accuracy, marker="o", color="#0891b2", linewidth=2)
    ax.axhline(0.25, color="#94a3b8", linestyle="--", linewidth=1, label="random (4 classes)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("SpectraCNN accuracy vs SNR")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    path: str | Path,
    title: str = "SpectraCNN confusion matrix",
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(array, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted modulation")
    ax.set_ylabel("True modulation")
    ax.set_title(title)
    threshold = array.max() / 2 if array.size else 0
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = int(array[row, column])
            color = "white" if value > threshold else "black"
            ax.text(column, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
