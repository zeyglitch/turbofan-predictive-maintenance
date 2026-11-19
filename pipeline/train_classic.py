"""Train the baseline and Ridge, evaluate them on the validation engines and save them."""

import json

import joblib
import numpy as np
import pandas as pd

from turbofan.classic_models import train_baseline, train_ridge
from turbofan.config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, load_params
from turbofan.features import feature_columns
from turbofan.metrics import alert_metrics, regression_report


def evaluate(y_true, y_pred, cap: int, threshold: float) -> dict:
    """Bound predictions to [0, cap], then return error and alert measures."""
    y_pred = np.clip(y_pred, 0, cap)
    return regression_report(y_true, y_pred) | alert_metrics(y_true, y_pred, threshold)


def main() -> None:
    params = load_params()
    sensors = params["prepare"]["sensors"]
    cap = params["prepare"]["rul_cap"]
    threshold = params["alert"]["threshold"]
    features = feature_columns(sensors)

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    y_train, y_val = train["rul_clipped"], val["rul_clipped"]
    print(f"train: {len(train)} rows, val: {len(val)} rows, {len(features)} features")

    baseline = train_baseline(y_train)
    ridge = train_ridge(train[features], y_train, params["ridge"])
    # same model on the current sensor values only, to measure what the rolling features add
    ridge_sensors_only = train_ridge(train[sensors], y_train, params["ridge"])

    results = {
        "baseline": evaluate(y_val, np.full(len(val), baseline), cap, threshold),
        "ridge": evaluate(y_val, ridge.predict(val[features]), cap, threshold),
        "ridge_sensors_only": evaluate(
            y_val, ridge_sensors_only.predict(val[sensors]), cap, threshold
        ),
    }

    print(f"{'model':<20}{'rmse':>8}{'mae':>8}{'nasa':>12}{'precision':>11}{'recall':>8}")
    for name, m in results.items():
        print(
            f"{name:<20}{m['rmse']:>8.2f}{m['mae']:>8.2f}{m['nasa_score']:>12.0f}"
            f"{m['precision']:>11.2f}{m['recall']:>8.2f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "baseline.json").write_text(json.dumps({"value": baseline}), encoding="utf-8")
    joblib.dump(ridge, MODELS_DIR / "ridge.joblib")
    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "val_classic.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {MODELS_DIR / 'ridge.joblib'} and {metrics_dir / 'val_classic.json'}")


if __name__ == "__main__":
    main()
