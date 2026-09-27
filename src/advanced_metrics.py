"""Evaluation metrics for nominal, ordinal, imbalanced, and calibrated outputs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(error)


def compute_metrics(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    expected_index = (
        probabilities * np.arange(probabilities.shape[1], dtype=float)[None, :]
    ).sum(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "class_index_mae": float(np.mean(np.abs(predictions - labels))),
        "expected_index_mae": float(np.mean(np.abs(expected_index - labels))),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, predictions, weights="quadratic")
        ),
        "ece_15_bin": expected_calibration_error(probabilities, labels, bins=15),
    }

