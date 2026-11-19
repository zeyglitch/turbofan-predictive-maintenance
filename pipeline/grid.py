"""Try a small grid of XGBoost settings, record each try in MLflow and print the best one.

Not part of the DVC pipeline: the best settings are copied into params.yaml by hand.
"""

import itertools

import pandas as pd

from turbofan.classic_models import predict, train_xgboost
from turbofan.config import FEATURES_DIR, load_params
from turbofan.features import feature_columns
from turbofan.metrics import evaluate_predictions
from turbofan.tracking import log_metrics, start_run

GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1]}


def main() -> None:
    params = load_params()
    cap = params["prepare"]["rul_cap"]
    threshold = params["alert"]["threshold"]
    features = feature_columns(params["prepare"]["sensors"])

    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    y_train, y_val = train["rul_clipped"], val["rul_clipped"]

    rows = []
    for max_depth, learning_rate in itertools.product(GRID["max_depth"], GRID["learning_rate"]):
        settings = params["xgboost"] | {"max_depth": max_depth, "learning_rate": learning_rate}
        booster = train_xgboost(train[features], y_train, val[features], y_val, settings)
        metrics = evaluate_predictions(y_val, predict(booster, val[features]), cap, threshold)
        trees = booster.best_iteration + 1
        name = f"xgboost_depth{max_depth}_lr{learning_rate}"
        with start_run(
            name,
            settings | {"rolling_window": params["features"]["rolling_window"]},
            tags={"grid": "xgboost"},
        ):
            log_metrics({f"val_{key}": value for key, value in metrics.items()} | {"trees": trees})
        rows.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "trees": trees,
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_recall": metrics["recall"],
            }
        )

    table = pd.DataFrame(rows).sort_values("val_rmse")
    print(table.to_string(index=False, float_format="{:.2f}".format))
    best = table.iloc[0]
    print(f"best: max_depth {best['max_depth']:.0f}, learning_rate {best['learning_rate']}")


if __name__ == "__main__":
    main()
