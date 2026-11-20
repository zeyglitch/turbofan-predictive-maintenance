"""Split conformal prediction: an interval around each prediction, from errors on unseen engines."""

import math

import numpy as np


def conformal_quantile(y_true, y_pred, coverage: float) -> float:
    """Return q such that about `coverage` of new errors |pred - true| are at most q.

    q is the k-th smallest calibration error, k = ceil((n + 1) * coverage), or the largest one
    when k > n.
    """
    errors = np.sort(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))
    n = len(errors)
    k = math.ceil((n + 1) * coverage)
    # not np.quantile: its interpolation rules would give another value than the k-th error
    return float(errors[min(k, n) - 1])


def prediction_interval(y_pred, q: float, rul_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [pred - q, pred + q], bounded to [0, rul_cap]."""
    y_pred = np.asarray(y_pred, dtype=float)
    return np.clip(y_pred - q, 0, rul_cap), np.clip(y_pred + q, 0, rul_cap)


def empirical_coverage(y_true, low, high) -> float:
    """Share of true values that fall inside their interval."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= low) & (y_true <= high)))
