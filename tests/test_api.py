import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from turbofan.bundle import save_bundle
from turbofan.classic_models import train_ridge
from turbofan.config import MODELS_DIR, REPORTS_DIR
from turbofan.data import load_raw
from turbofan.features import add_rolling_features, feature_columns
from turbofan.labels import add_rul
from turbofan.preprocessing import fit_scaler, scale_sensors
from turbofan.retraining import champion_replaced

SENSORS = ["s_2", "s_11", "s_3"]
RESPONSE_KEYS = {"unit_id", "cycle", "rul", "rul_low", "rul_high", "alert", "model_version"}
CHAMPION_DIR = MODELS_DIR / "champion"


def toy_bundle_on_disk(small_fleet, directory) -> None:
    """A Ridge bundle trained on the synthetic fleet, written like the real models/champion."""
    fleet = add_rul(small_fleet, 125)
    scaler = fit_scaler(fleet, SENSORS)
    features = add_rolling_features(scale_sensors(fleet, scaler, SENSORS), SENSORS, 10)
    model = train_ridge(features[feature_columns(SENSORS)], features["rul_clipped"], {})
    metadata = {
        "model_params": {},
        "subsets": ["FD001"],
        "sensors": SENSORS,
        "window_length": 30,
        "rolling_window": 10,
        "rul_cap": 125,
        "conformal_q": 15.0,
        "alert_threshold": 30,
        "val_rmse": 20.0,
        "test_rmse": 21.0,
        "created_at": "2026-11-20T10:00:00",
        "git_commit": "abc1234",
        "mlflow_run_id": "toy",
        "registry_version": "1",
    }
    save_bundle(directory, "ridge", model, scaler, metadata)


def request_body(engine: pd.DataFrame) -> dict:
    """The JSON body of POST /predict for the rows of one engine."""
    cycles = engine.drop(columns="unit_id").to_dict(orient="records")
    return {"unit_id": int(engine["unit_id"].iloc[0]), "cycles": cycles}


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "logs" / "predictions.jsonl"


@pytest.fixture
def client(small_fleet, tmp_path, log_path, monkeypatch):
    toy_bundle_on_disk(small_fleet, tmp_path / "champion")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "champion"))
    monkeypatch.setenv("PREDICTION_LOG", str(log_path))
    with TestClient(app) as client:  # the with block runs the start-up, which loads the bundle
        yield client


@pytest.fixture
def engine_40_cycles(small_fleet):
    return small_fleet[small_fleet["unit_id"] == 1].head(40)


def test_health_says_the_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_describes_the_served_bundle(client):
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "ridge"
    assert body["sensors"] == SENSORS
    assert body["model_version"] == "1"


def test_predict_returns_the_seven_fields(client, engine_40_cycles):
    response = client.post("/predict", json=request_body(engine_40_cycles))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["unit_id"] == 1
    assert body["cycle"] == 40
    assert 0 <= body["rul_low"] <= body["rul"] <= body["rul_high"] <= 125
    assert body["model_version"] == "1"


def test_an_empty_history_is_refused(client):
    response = client.post("/predict", json={"unit_id": 1, "cycles": []})
    assert response.status_code == 422


def test_a_missing_sensor_is_refused_with_its_name(client, engine_40_cycles):
    body = request_body(engine_40_cycles)
    del body["cycles"][5]["s_3"]
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    # the error says where: in the body, cycle number 5 of the list, field s_3
    assert response.json()["detail"][0]["loc"] == ["body", "cycles", 5, "s_3"]


def test_a_text_value_is_refused(client, engine_40_cycles):
    body = request_body(engine_40_cycles)
    body["cycles"][0]["s_2"] = "high"
    response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_each_prediction_adds_one_json_line_to_the_log(client, engine_40_cycles, log_path):
    client.post("/predict", json=request_body(engine_40_cycles))
    client.post("/predict", json=request_body(engine_40_cycles.head(20)))
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[1])
    assert record["cycle"] == 20
    assert record["s_21"] == pytest.approx(engine_40_cycles["s_21"].iloc[19])
    assert {"timestamp", "rul", "rul_low", "rul_high", "alert", "model_version"} <= set(record)


def test_no_log_file_without_prediction_log(small_fleet, tmp_path, engine_40_cycles, monkeypatch):
    toy_bundle_on_disk(small_fleet, tmp_path / "champion")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "champion"))
    monkeypatch.delenv("PREDICTION_LOG", raising=False)
    monkeypatch.chdir(tmp_path)  # so that a log written by mistake would land here
    with TestClient(app) as client:
        assert client.post("/predict", json=request_body(engine_40_cycles)).status_code == 200
    assert sorted(path.name for path in tmp_path.iterdir()) == ["champion"]


@pytest.mark.needs_data
@pytest.mark.skipif(
    not (CHAMPION_DIR / "metadata.json").exists(), reason="no champion, run dvc repro"
)
@pytest.mark.skipif(
    champion_replaced(),
    reason="the retraining flow replaced models/champion; dvc checkout models/champion restores it",
)
def test_api_gives_the_predictions_of_the_evaluation(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(CHAMPION_DIR))
    monkeypatch.delenv("PREDICTION_LOG", raising=False)
    evaluation = pd.read_parquet(REPORTS_DIR / "predictions_test.parquet")
    with TestClient(app) as client:
        champion = client.get("/model").json()["model_type"]
        expected = evaluation[evaluation["model"] == champion].set_index("unit_id")["rul_pred"]
        for unit_id, engine in load_raw("FD001", "test").groupby("unit_id"):
            rul = client.post("/predict", json=request_body(engine)).json()["rul"]
            assert rul == pytest.approx(expected[unit_id], abs=1e-3)
