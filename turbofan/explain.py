"""Which inputs weigh in the predictions: SHAP values for XGBoost, permutation for the networks."""

import numpy as np
import pandas as pd
import shap

from turbofan.metrics import rmse


def shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """SHAP value of every feature for every row of X, for a tree model such as XGBoost.

    For each row, base value + sum of its SHAP values = the model's prediction.
    """
    return shap.TreeExplainer(model)(X)


def shap_importance(explanation: shap.Explanation) -> pd.Series:
    """Global importance: mean |SHAP value| of each feature, in cycles, largest first."""
    importance = np.abs(explanation.values).mean(axis=0)
    return pd.Series(importance, index=explanation.feature_names).sort_values(ascending=False)


def importance_per_sensor(importance: pd.Series, sensors: list[str]) -> pd.Series:
    """Add up the importance of a sensor's value, rolling mean and rolling std."""
    per_sensor = {
        sensor: importance[[sensor, f"{sensor}_mean", f"{sensor}_std"]].sum() for sensor in sensors
    }
    return pd.Series(per_sensor).sort_values(ascending=False)


def permutation_importance(
    predict_fn, X: np.ndarray, y, sensors: list[str], n_repeats: int, seed: int
) -> pd.Series:
    """How much the RMSE rises when one sensor is shuffled between windows, averaged over repeats.

    X has shape (n_windows, length, n_sensors). Shuffling moves the whole history of a sensor to
    another window, which breaks its link with the target and keeps everything else.
    """
    rng = np.random.default_rng(seed)
    base_rmse = rmse(y, predict_fn(X))
    increases = {}
    for j, sensor in enumerate(sensors):
        scores = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[:, :, j] = X[rng.permutation(len(X)), :, j]
            scores.append(rmse(y, predict_fn(X_shuffled)) - base_rmse)
        increases[sensor] = float(np.mean(scores))
    return pd.Series(increases).sort_values(ascending=False)
