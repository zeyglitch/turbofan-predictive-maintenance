"""Explain XGBoost with SHAP values and the LSTM with permutation importance, on validation data."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import shap

from pipeline.evaluate import load_trained
from turbofan.config import FEATURES_DIR, PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.explain import (
    importance_per_sensor,
    permutation_importance,
    shap_importance,
    shap_values,
)
from turbofan.features import feature_columns
from turbofan.plots import plot_importance
from turbofan.training import predict_network
from turbofan.windows import make_windows

EXPLAIN_DIR = REPORTS_DIR / "explain"


def save_figure(figure, file_name: str) -> None:
    figure.savefig(EXPLAIN_DIR / file_name, dpi=120, bbox_inches="tight")
    plt.close(figure)


def explain_xgboost(params: dict) -> pd.Series:
    """SHAP values on a sample of validation rows; returns the importance of each feature."""
    sensors = params["prepare"]["sensors"]
    features = feature_columns(sensors)
    settings = params["explain"]
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    sample = val.sample(settings["sample_size"], random_state=settings["seed"])
    explanation = shap_values(load_trained("xgboost", params), sample[features])

    importance = shap_importance(explanation)
    importance.rename("mean_abs_shap").to_csv(EXPLAIN_DIR / "xgboost_shap_importance.csv")
    values = pd.DataFrame(explanation.values, columns=features)
    values.insert(0, "cycle", sample["cycle"].to_numpy())
    values.insert(0, "unit_id", sample["unit_id"].to_numpy())
    values.to_parquet(EXPLAIN_DIR / "xgboost_shap_values.parquet", index=False)

    save_figure(
        plot_importance(importance.head(15), "XGBoost, 15 most important features", "mean |SHAP|"),
        "xgboost_shap_bar.png",
    )
    # one dot per row and feature: position = SHAP value, colour = value of the feature
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    save_figure(plt.gcf(), "xgboost_shap_beeswarm.png")
    return importance


def explain_one_prediction(params: dict) -> dict:
    """Local explanation: the first validation engine, 20 cycles before failure."""
    features = feature_columns(params["prepare"]["sensors"])
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")
    engine = val[val["unit_id"] == val["unit_id"].min()]
    row = engine[engine["rul"] == 20]
    explanation = shap_values(load_trained("xgboost", params), row[features])
    shap.plots.waterfall(explanation[0], max_display=12, show=False)
    save_figure(plt.gcf(), "xgboost_shap_one_prediction.png")
    return {
        "unit_id": int(row["unit_id"].iloc[0]),
        "cycle": int(row["cycle"].iloc[0]),
        "true_rul": 20,
        "base_value": float(explanation.base_values[0]),
        "prediction": float(explanation.base_values[0] + explanation.values[0].sum()),
    }


def explain_lstm(params: dict) -> pd.Series:
    """Permutation importance of each sensor for the LSTM, on every validation window."""
    sensors = params["prepare"]["sensors"]
    cap = params["prepare"]["rul_cap"]
    settings = params["explain"]
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    X, y, _ = make_windows(val, sensors, params["windows"]["length"], every_cycle=True)
    model = load_trained("lstm", params)

    importance = permutation_importance(
        lambda windows: predict_network(model, windows, cap),
        X,
        y,
        sensors,
        settings["permutation_repeats"],
        settings["seed"],
    )
    importance.rename("rmse_increase").to_csv(EXPLAIN_DIR / "lstm_permutation_importance.csv")
    save_figure(
        plot_importance(importance, "LSTM, permutation importance", "validation RMSE increase"),
        "lstm_permutation.png",
    )
    return importance


def main() -> None:
    params = load_params()
    sensors = params["prepare"]["sensors"]
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

    xgb_importance = explain_xgboost(params)
    one_prediction = explain_one_prediction(params)
    lstm_importance = explain_lstm(params)

    # the two methods side by side, sensor by sensor (rank 1 = most important)
    ranking = pd.DataFrame(
        {
            "xgboost_shap": importance_per_sensor(xgb_importance, sensors),
            "lstm_permutation": lstm_importance,
        }
    )
    ranking["xgboost_rank"] = ranking["xgboost_shap"].rank(ascending=False).astype(int)
    ranking["lstm_rank"] = ranking["lstm_permutation"].rank(ascending=False).astype(int)
    ranking = ranking.sort_values("lstm_rank")
    ranking.to_csv(EXPLAIN_DIR / "sensor_ranking.csv", index_label="sensor")
    (EXPLAIN_DIR / "one_prediction.json").write_text(
        json.dumps(one_prediction, indent=2), encoding="utf-8", newline="\n"
    )
    print(ranking.round(2).to_string())
    print(f"one prediction: {one_prediction}")
    print(f"wrote tables and figures to {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
