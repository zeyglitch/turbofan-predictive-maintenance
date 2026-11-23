"""Score the six models on the 100 test engines, with intervals calibrated on the calib engines.

The test set is read here for the first time: nothing in this script chooses or tunes a model.
The true remaining life is capped at 125, like the training target: a test engine with 180
cycles left counts as 125. rmse_uncapped compares the same predictions with the true value.
"""

import json

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

from turbofan.classic_models import predict
from turbofan.config import FEATURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.conformal import conformal_quantile, empirical_coverage, prediction_interval
from turbofan.features import feature_columns, last_row_per_unit
from turbofan.metrics import evaluate_predictions, rmse
from turbofan.networks import build_network
from turbofan.training import predict_network
from turbofan.windows import last_window_per_unit, make_windows

MODELS = ["baseline", "ridge", "random_forest", "xgboost", "lstm", "cnn"]
NETWORKS = ["lstm", "cnn"]


def load_trained(name: str, params: dict):
    """Read back a model written by train_classic or train_deep."""
    if name == "baseline":
        return json.loads((MODELS_DIR / "baseline.json").read_text(encoding="utf-8"))["value"]
    if name == "xgboost":
        booster = XGBRegressor()
        booster.load_model(MODELS_DIR / "xgboost.json")
        return booster
    if name in NETWORKS:
        model = build_network(name, len(params["prepare"]["sensors"]), params[name])
        model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt"))
        return model
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def predict_part(name: str, model, part: str, params: dict, last_only: bool):
    """Return the true and predicted rul_clipped for a part (calib or test).

    last_only=False: every row of every engine. last_only=True: the last cycle of each engine,
    sorted by unit_id, which is how the official test set is scored.
    """
    sensors = params["prepare"]["sensors"]
    cap = params["prepare"]["rul_cap"]
    if name in NETWORKS:
        df = pd.read_parquet(PROCESSED_DIR / f"{part}.parquet")
        length = params["windows"]["length"]
        if last_only:
            X, _ = last_window_per_unit(df, sensors, length)
            y = last_row_per_unit(df)["rul_clipped"].to_numpy()
        else:
            X, y, _ = make_windows(df, sensors, length, every_cycle=True)
        return y, predict_network(model, X, cap)
    df = pd.read_parquet(FEATURES_DIR / f"{part}.parquet")
    if last_only:
        df = last_row_per_unit(df)
    y_pred = np.clip(predict(model, df[feature_columns(sensors)]), 0, cap)
    return df["rul_clipped"].to_numpy(), y_pred


def main() -> None:
    params = load_params()
    cap = params["prepare"]["rul_cap"]
    coverage = params["conformal"]["coverage"]
    threshold = params["alert"]["threshold"]
    last_test_rows = last_row_per_unit(pd.read_parquet(PROCESSED_DIR / "test.parquet"))
    unit_ids = last_test_rows["unit_id"]
    # the true remaining life, not capped: the models never predict more than the cap, so the
    # engines still far from failure count as errors here
    y_test_uncapped = last_test_rows["rul"].to_numpy()

    results, predictions = {}, []
    header = f"{'model':<15}{'rmse':>7}{'uncapped':>10}{'mae':>7}{'nasa':>9}{'recall':>8}{'q':>7}"
    print(header + f"{'coverage':>10}")
    for name in MODELS:
        model = load_trained(name, params)
        # q comes from engines that took part in no training and no choice
        y_calib, calib_pred = predict_part(name, model, "calib", params, last_only=False)
        q = conformal_quantile(y_calib, calib_pred, coverage)
        y_test, test_pred = predict_part(name, model, "test", params, last_only=True)
        low, high = prediction_interval(test_pred, q, cap)
        metrics = evaluate_predictions(y_test, test_pred, cap, threshold)
        results[name] = metrics | {
            "rmse_uncapped": rmse(y_test_uncapped, test_pred),
            "conformal_q": q,
            "coverage": empirical_coverage(y_test, low, high),
        }
        predictions.append(
            pd.DataFrame(
                {
                    "model": name,
                    "unit_id": unit_ids.to_numpy(),
                    "rul_true": y_test,
                    "rul_true_uncapped": y_test_uncapped,
                    "rul_pred": test_pred,
                    "rul_low": low,
                    "rul_high": high,
                }
            )
        )
        r = results[name]
        print(
            f"{name:<15}{r['rmse']:>7.2f}{r['rmse_uncapped']:>10.2f}{r['mae']:>7.2f}"
            f"{r['nasa_score']:>9.1f}"
            f"{r['recall']:>8.2f}{q:>7.1f}{r['coverage']:>10.2f}"
        )

    pd.concat(predictions).to_parquet(REPORTS_DIR / "predictions_test.parquet", index=False)
    # newline="\n": the same bytes on Windows and Linux, so the fingerprint in dvc.lock matches
    (REPORTS_DIR / "metrics" / "test.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"wrote {REPORTS_DIR / 'metrics' / 'test.json'} and predictions_test.parquet")


if __name__ == "__main__":
    main()
