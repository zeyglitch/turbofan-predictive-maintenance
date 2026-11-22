import numpy as np
import pytest

from turbofan.bundle import load_bundle
from turbofan.classic_models import predict, train_ridge
from turbofan.config import MODELS_DIR
from turbofan.data import load_raw, load_test_rul
from turbofan.features import add_rolling_features, feature_columns
from turbofan.labels import add_rul, clip_rul
from turbofan.metrics import rmse
from turbofan.networks import LSTMRegressor
from turbofan.predict import predict_engine
from turbofan.preprocessing import fit_scaler, scale_sensors
from turbofan.retraining import champion_replaced
from turbofan.training import set_seed

SENSORS = ["s_2", "s_11", "s_3"]
KEYS = {"rul", "rul_low", "rul_high", "alert"}
CHAMPION_DIR = MODELS_DIR / "champion"


def toy_bundle(small_fleet, model_type: str) -> dict:
    """A bundle in memory, trained on the synthetic fleet: Ridge or an untrained small LSTM."""
    fleet = add_rul(small_fleet, 125)
    scaler = fit_scaler(fleet, SENSORS)
    metadata = {
        "model_type": model_type,
        "sensors": SENSORS,
        "window_length": 30,
        "rolling_window": 10,
        "rul_cap": 125,
        "conformal_q": 15.0,
        "alert_threshold": 30,
    }
    if model_type == "ridge":
        features = add_rolling_features(scale_sensors(fleet, scaler, SENSORS), SENSORS, 10)
        model = train_ridge(features[feature_columns(SENSORS)], features["rul_clipped"], {})
    else:
        set_seed(0)
        model = LSTMRegressor(n_sensors=len(SENSORS), hidden_size=4, num_layers=1, dropout=0.0)
    return {"model": model, "scaler": scaler, "metadata": metadata}


@pytest.mark.parametrize("model_type", ["ridge", "lstm"])
def test_prediction_has_an_interval_around_it_and_an_alert(small_fleet, model_type):
    engine = small_fleet[small_fleet["unit_id"] == 1]
    result = predict_engine(toy_bundle(small_fleet, model_type), engine)
    assert set(result) == KEYS
    assert 0 <= result["rul_low"] <= result["rul"] <= result["rul_high"] <= 125
    assert result["alert"] == (result["rul"] <= 30)


@pytest.mark.parametrize("model_type", ["ridge", "lstm"])
def test_a_history_shorter_than_the_window_still_gets_a_prediction(small_fleet, model_type):
    first_10_cycles = small_fleet[small_fleet["unit_id"] == 1].head(10)
    result = predict_engine(toy_bundle(small_fleet, model_type), first_10_cycles)
    assert set(result) == KEYS
    assert 0 <= result["rul"] <= 125


def test_ridge_prediction_matches_the_training_preparation(small_fleet):
    bundle = toy_bundle(small_fleet, "ridge")
    engine = small_fleet[small_fleet["unit_id"] == 3]
    # the same steps written out by hand: scale, rolling features, last row, predict
    scaled = scale_sensors(engine, bundle["scaler"], SENSORS)
    last_row = add_rolling_features(scaled, SENSORS, 10).iloc[[-1]]
    expected = np.clip(predict(bundle["model"], last_row[feature_columns(SENSORS)])[0], 0, 125)
    assert predict_engine(bundle, engine)["rul"] == pytest.approx(expected)


@pytest.mark.needs_data
@pytest.mark.skipif(
    not (CHAMPION_DIR / "metadata.json").exists(), reason="no champion, run dvc repro"
)
@pytest.mark.skipif(
    champion_replaced(),
    reason="the retraining flow replaced models/champion; dvc checkout models/champion restores it",
)
def test_champion_on_raw_test_engines_gives_the_rmse_of_the_evaluation():
    bundle = load_bundle(CHAMPION_DIR)
    raw = load_raw("FD001", "test")
    # the official truth at the last cycle of each engine, capped like the training target
    truth = clip_rul(load_test_rul("FD001"), bundle["metadata"]["rul_cap"])
    rul = [predict_engine(bundle, engine)["rul"] for _, engine in raw.groupby("unit_id")]
    # raw history -> scaling -> windows -> model: the path of the API, not the one of evaluate
    assert rmse(truth.sort_index(), rul) == pytest.approx(bundle["metadata"]["test_rmse"], abs=0.01)


def test_rows_out_of_order_give_the_same_prediction(small_fleet):
    bundle = toy_bundle(small_fleet, "ridge")
    engine = small_fleet[small_fleet["unit_id"] == 2]
    shuffled = engine.sample(frac=1, random_state=0)
    assert predict_engine(bundle, shuffled)["rul"] == pytest.approx(
        predict_engine(bundle, engine)["rul"]
    )
