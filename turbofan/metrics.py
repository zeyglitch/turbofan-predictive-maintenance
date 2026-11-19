"""Error measures in cycles, and the quality of the derived "less than N cycles left" alert."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in cycles. Large errors weigh more than small ones."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in cycles: how far off the prediction is on average."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(d)))


def nasa_score(y_true, y_pred) -> float:
    """Sum over engines of exp(-d/13) - 1 if d < 0, exp(d/10) - 1 otherwise, with d = pred - true.

    Predicting too much remaining life (d > 0) is punished harder than predicting too little.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(penalties))


def regression_report(y_true, y_pred) -> dict:
    """Return the three error measures in one dictionary."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def evaluate_predictions(y_true, y_pred, cap: int, threshold: float) -> dict:
    """Bound predictions to [0, cap], then return the error measures and the alert measures."""
    # the target never leaves [0, cap]; a linear model can, for instance predict -142 cycles
    y_pred = np.clip(y_pred, 0, cap)
    return regression_report(y_true, y_pred) | alert_metrics(y_true, y_pred, threshold)


def alert_metrics(y_true, y_pred, threshold: float) -> dict:
    """Treat "rul <= threshold" as an alert and count good and bad alerts.

    precision: share of raised alerts that were real; recall: share of real ones that were raised.
    """
    true_alert = np.asarray(y_true) <= threshold
    predicted_alert = np.asarray(y_pred) <= threshold
    tp = int(np.sum(true_alert & predicted_alert))
    fp = int(np.sum(~true_alert & predicted_alert))
    fn = int(np.sum(true_alert & ~predicted_alert))
    tn = int(np.sum(~true_alert & ~predicted_alert))
    # no alert raised (or none needed) gives 0 instead of a division by zero
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}
