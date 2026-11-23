"""Try a small grid of settings for one model, record each try in MLflow and print the best one.

Run with `python -m pipeline.grid xgboost` (or lstm, random_forest, cnn).
Not part of the DVC pipeline: the best settings are copied into params.yaml by hand. Every grid
is scored on the validation engines; the training RMSE is printed too, to see the overfitting.
"""

import argparse
import itertools

import pandas as pd

from pipeline.train_deep import fit_network
from turbofan.classic_models import predict, train_random_forest, train_xgboost
from turbofan.config import FEATURES_DIR, PROCESSED_DIR, load_params
from turbofan.features import feature_columns
from turbofan.metrics import evaluate_predictions, rmse
from turbofan.tracking import log_metrics, start_run

GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1]}
LSTM_GRID = {"hidden_size": [32, 64, 128], "window_length": [20, 30, 50]}
# against the overfitting seen in the audit: shallower trees, more rows in each leaf
FOREST_GRID = {"max_depth": [6, 8, 12], "min_samples_leaf": [5, 20, 50]}
# against the overfitting of the CNN: dropout, and fewer filters
CNN_GRID = {"dropout": [0.0, 0.3], "channels": [[32, 64, 64], [16, 32, 32]]}


def grid_xgboost(params: dict) -> pd.DataFrame:
    """Every pair of max_depth and learning_rate, the other settings taken from params.yaml."""
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
    return pd.DataFrame(rows)


def grid_random_forest(params: dict) -> pd.DataFrame:
    """Every pair of max_depth and min_samples_leaf, the other settings taken from params.yaml."""
    cap = params["prepare"]["rul_cap"]
    threshold = params["alert"]["threshold"]
    features = feature_columns(params["prepare"]["sensors"])
    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    y_train, y_val = train["rul_clipped"], val["rul_clipped"]

    rows = []
    pairs = itertools.product(FOREST_GRID["max_depth"], FOREST_GRID["min_samples_leaf"])
    for max_depth, min_samples_leaf in pairs:
        settings = params["random_forest"] | {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
        forest = train_random_forest(train[features], y_train, settings)
        val_pred = predict(forest, val[features]).clip(0, cap)
        metrics = evaluate_predictions(y_val, val_pred, cap, threshold)
        train_rmse = rmse(y_train, predict(forest, train[features]).clip(0, cap))
        name = f"random_forest_depth{max_depth}_leaf{min_samples_leaf}"
        with start_run(name, settings, tags={"grid": "random_forest"}):
            log_metrics({f"val_{key}": value for key, value in metrics.items()})
            log_metrics({"train_rmse": train_rmse})
        rows.append(
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "train_rmse": train_rmse,
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_recall": metrics["recall"],
            }
        )
    return pd.DataFrame(rows)


def grid_cnn(params: dict) -> pd.DataFrame:
    """Every pair of dropout and channels, with the window length of params.yaml."""
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    length = params["windows"]["length"]

    rows = []
    for dropout, channels in itertools.product(CNN_GRID["dropout"], CNN_GRID["channels"]):
        cnn_params = params["cnn"] | {"dropout": dropout, "channels": channels}
        name = f"cnn_dropout{dropout}_channels{'-'.join(map(str, channels))}"
        _, metrics, _ = fit_network(
            params, train, val, "cnn", cnn_params, length, name, {"grid": "cnn"}
        )
        rows.append(
            {
                "dropout": dropout,
                "channels": channels,
                "best_epoch": metrics["best_epoch"],
                "train_rmse": metrics["train_rmse"],
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_recall": metrics["recall"],
            }
        )
    return pd.DataFrame(rows)


def grid_lstm(params: dict) -> pd.DataFrame:
    """First the hidden size with the default window, then the window with the best size."""
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    default_length = params["windows"]["length"]
    tags = {"grid": "lstm"}

    def try_one(hidden_size: int, length: int) -> dict:
        lstm_params = params["lstm"] | {"hidden_size": hidden_size}
        name = f"lstm_hidden{hidden_size}_window{length}"
        _, metrics, _ = fit_network(params, train, val, "lstm", lstm_params, length, name, tags)
        return {
            "hidden_size": hidden_size,
            "window_length": length,
            "best_epoch": metrics["best_epoch"],
            "val_rmse": metrics["rmse"],
            "val_mae": metrics["mae"],
            "val_recall": metrics["recall"],
        }

    rows = [try_one(size, default_length) for size in LSTM_GRID["hidden_size"]]
    best_size = min(rows, key=lambda row: row["val_rmse"])["hidden_size"]
    # the default window was already tried with the best size in the first round
    other_lengths = [length for length in LSTM_GRID["window_length"] if length != default_length]
    rows += [try_one(best_size, length) for length in other_lengths]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grids = {
        "xgboost": grid_xgboost,
        "lstm": grid_lstm,
        "random_forest": grid_random_forest,
        "cnn": grid_cnn,
    }
    parser.add_argument("model", choices=list(grids))
    model = parser.parse_args().model

    params = load_params()
    table = grids[model](params)
    table = table.sort_values("val_rmse")
    print(table.to_string(index=False, float_format="{:.2f}".format))
    scores = ["train_rmse", "val_rmse", "val_mae", "val_recall"]
    print("best:", table.iloc[0].drop(scores, errors="ignore").to_dict())


if __name__ == "__main__":
    main()
