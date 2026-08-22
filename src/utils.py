from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def classification_metrics(y_true, y_pred, probabilities):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
    }


def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probabilities >= bins[i]) & (probabilities <= bins[i + 1])
        else:
            mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])

        if mask.sum() == 0:
            continue

        bin_confidence = probabilities[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def selective_prediction_analysis(y_true, probabilities, thresholds):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    rows = []
    for threshold in thresholds:
        accept = confidence >= threshold
        n_total = len(y_true)
        n_predicted = int(accept.sum())
        n_deferred = n_total - n_predicted

        coverage = n_predicted / n_total
        deferral_rate = n_deferred / n_total

        if n_predicted > 0:
            selective_accuracy = accuracy_score(y_true[accept], predictions[accept])
            selective_error = 1.0 - selective_accuracy
            selective_recall = recall_score(
                y_true[accept], predictions[accept], zero_division=0
            )
        else:
            selective_accuracy = np.nan
            selective_error = np.nan
            selective_recall = np.nan

        rows.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "deferral_rate": deferral_rate,
                "selective_accuracy": selective_accuracy,
                "selective_error": selective_error,
                "selective_recall": selective_recall,
                "predicted_cases": n_predicted,
                "deferred_cases": n_deferred,
            }
        )

    return pd.DataFrame(rows)


def decision_utility(
    y_true,
    probabilities,
    threshold,
    correct_reward=1.0,
    wrong_cost=-2.0,
    defer_cost=-0.25,
):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    accept = confidence >= threshold

    utility = np.full(len(y_true), defer_cost, dtype=float)
    utility[accept & (predictions == y_true)] = correct_reward
    utility[accept & (predictions != y_true)] = wrong_cost

    return float(utility.mean())
