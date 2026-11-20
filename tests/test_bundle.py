import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from turbofan.bundle import METADATA_KEYS, load_bundle, save_bundle
from turbofan.networks import LSTMRegressor
from turbofan.training import predict_network, set_seed

SENSORS = ["s_2", "s_11", "s_3"]


def toy_metadata(model_params: dict) -> dict:
    """Every key the bundle requires, with toy values."""
    return {
        "model_params": model_params,
        "subsets": ["FD001"],
        "sensors": SENSORS,
        "window_length": 30,
        "rolling_window": 30,
        "rul_cap": 125,
        "conformal_q": 20.0,
        "alert_threshold": 30,
        "val_rmse": 15.0,
        "test_rmse": 16.0,
        "created_at": "2026-11-19T12:00:00",
        "git_commit": "abc1234",
        "mlflow_run_id": "run",
        "registry_version": "1",
    }


def toy_scaler() -> StandardScaler:
    return StandardScaler().fit(np.random.default_rng(0).normal(size=(50, len(SENSORS))))


def test_ridge_gives_the_same_predictions_after_saving_and_loading(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    ridge = Ridge(alpha=1.0).fit(X, y)
    save_bundle(tmp_path, "ridge", ridge, toy_scaler(), toy_metadata({"alpha": 1.0}))

    bundle = load_bundle(tmp_path)
    np.testing.assert_allclose(bundle["model"].predict(X), ridge.predict(X), atol=1e-5)


def test_xgboost_gives_the_same_predictions_after_saving_and_loading(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    booster = XGBRegressor(n_estimators=5, max_depth=2).fit(X, y)
    save_bundle(tmp_path, "xgboost", booster, toy_scaler(), toy_metadata({"max_depth": 2}))

    bundle = load_bundle(tmp_path)
    np.testing.assert_allclose(bundle["model"].predict(X), booster.predict(X), atol=1e-5)


def test_lstm_gives_the_same_predictions_after_saving_and_loading(tmp_path):
    params = {"hidden_size": 4, "num_layers": 2, "dropout": 0.2}
    set_seed(0)
    lstm = LSTMRegressor(n_sensors=len(SENSORS), **params)
    X = np.random.default_rng(0).normal(size=(6, 30, len(SENSORS))).astype(np.float32)
    save_bundle(tmp_path, "lstm", lstm, toy_scaler(), toy_metadata(params))

    bundle = load_bundle(tmp_path)
    np.testing.assert_allclose(
        predict_network(bundle["model"], X, 125), predict_network(lstm, X, 125), atol=1e-5
    )


def test_metadata_is_read_back_with_every_key(tmp_path):
    save_bundle(tmp_path, "baseline", 88.0, toy_scaler(), toy_metadata({}))

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert set(METADATA_KEYS) <= set(metadata)
    assert metadata["model_type"] == "baseline"
    assert load_bundle(tmp_path)["model"] == 88.0


def test_saving_refuses_incomplete_metadata(tmp_path):
    metadata = toy_metadata({})
    del metadata["conformal_q"]
    with pytest.raises(ValueError, match="conformal_q"):
        save_bundle(tmp_path, "baseline", 88.0, toy_scaler(), metadata)
