"""Train the classic models, evaluate them on the validation engines, track and save them."""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from turbofan.classic_models import (
    predict,
    train_baseline,
    train_random_forest,
    train_ridge,
    train_xgboost,
)
from turbofan.config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, load_params
from turbofan.features import feature_columns
from turbofan.metrics import evaluate_predictions, rmse
from turbofan.tracking import log_figure, log_metrics, log_model, start_run


def plot_predictions(y_true, y_pred, title: str):
    """Scatter plot of predicted against true remaining life; a perfect model sits on the line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.3)
    ax.plot([0, 125], [0, 125], color="black", linewidth=1)
    ax.set_xlabel("true rul_clipped")
    ax.set_ylabel("predicted (bounded to [0, 125])")
    ax.set_title(title)
    return fig


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
    forest = train_random_forest(train[features], y_train, params["random_forest"])
    booster = train_xgboost(train[features], y_train, val[features], y_val, params["xgboost"])

    # name, model, input columns, settings to record, MLflow flavour (None: nothing to attach)
    runs = [
        ("baseline", baseline, features, {}, None),
        ("ridge", ridge, features, params["ridge"], "sklearn"),
        ("ridge_sensors_only", ridge_sensors_only, sensors, params["ridge"], "sklearn"),
        ("random_forest", forest, features, params["random_forest"], "sklearn"),
        ("xgboost", booster, features, params["xgboost"], "xgboost"),
    ]
    common = {
        "subset": params["data"]["subset"],
        "rul_cap": cap,
        "rolling_window": params["features"]["rolling_window"],
        "split_seed": params["split"]["seed"],
    }

    results = {}
    print(f"{'model':<20}{'train rmse':>11}{'val rmse':>9}{'val mae':>8}{'recall':>8}")
    for name, model, columns, settings, flavour in runs:
        val_pred = predict(model, val[columns]).clip(0, cap)
        metrics = evaluate_predictions(y_val, val_pred, cap, threshold)
        train_rmse = rmse(y_train, predict(model, train[columns]).clip(0, cap))
        results[name] = metrics | {"train_rmse": train_rmse}
        with start_run(name, common | settings | {"n_inputs": len(columns)}):
            log_metrics({f"val_{key}": value for key, value in metrics.items()})
            log_metrics({"train_rmse": train_rmse})
            figure = plot_predictions(y_val, val_pred, f"{name}, validation")
            log_figure(figure, "pred_vs_true_val.png")
            plt.close(figure)
            if flavour is not None:
                log_model(model, flavour, input_example=val[columns].head(5))
        print(
            f"{name:<20}{train_rmse:>11.2f}{metrics['rmse']:>9.2f}"
            f"{metrics['mae']:>8.2f}{metrics['recall']:>8.2f}"
        )
    print(f"xgboost stopped at {booster.best_iteration + 1} trees out of {booster.n_estimators}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "baseline.json").write_text(json.dumps({"value": baseline}), encoding="utf-8")
    joblib.dump(ridge, MODELS_DIR / "ridge.joblib")
    joblib.dump(forest, MODELS_DIR / "random_forest.joblib")
    booster.save_model(MODELS_DIR / "xgboost.json")
    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "val_classic.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote 4 models to {MODELS_DIR} and {metrics_dir / 'val_classic.json'}")


if __name__ == "__main__":
    main()
