"""Each page of the dashboard loads without an exception, with or without the pipeline's files.

AppTest runs the app in the test process, without a browser or a server.
"""

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import api_client
from turbofan.config import COLUMNS, ROOT, load_params
from turbofan.features import add_rolling_features
from turbofan.labels import add_rul
from turbofan.preprocessing import fit_scaler, scale_sensors

APP = str(ROOT / "dashboard" / "app.py")
PAGES = [
    "views/overview.py",
    "views/data_explorer.py",
    "views/preprocessing.py",
    "views/models.py",
    "views/engine_view.py",
    "views/explainability.py",
    "views/live_monitoring.py",
    "views/retraining.py",
]
LIVE = "views/live_monitoring.py"
RETRAINING = "views/retraining.py"
GLOSSARY = "views/glossary.py"
MODELS = ["baseline", "ridge", "random_forest", "xgboost", "lstm", "cnn"]
PARTS = {"train": [1, 2, 3, 4], "val": [5, 6], "calib": [7, 8]}


def write_json(path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


def scores(rmse: float) -> dict:
    """The keys the pipeline writes for one model in the metrics files."""
    return {
        "rmse": rmse,
        "mae": rmse * 0.8,
        "nasa_score": rmse * 30,
        "tp": 3,
        "fp": 1,
        "fn": 1,
        "tn": 3,
        "precision": 0.75,
        "recall": 0.75,
        "coverage": 0.9,
        "best_epoch": 4,
        "epochs_run": 6,
    }


@pytest.fixture
def toy_root(tmp_path, small_fleet):
    """A project folder holding small versions of every file the pages read."""
    shutil.copy(ROOT / "params.yaml", tmp_path / "params.yaml")
    shutil.copy(ROOT / "dvc.yaml", tmp_path / "dvc.yaml")
    params = load_params(tmp_path / "params.yaml")
    sensors = params["prepare"]["sensors"]

    # the synthetic fleet serves as both the training and the test file
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for part in ("train", "test"):
        small_fleet[COLUMNS].to_csv(
            raw_dir / f"{part}_FD001.txt", sep=" ", header=False, index=False
        )

    fleet = add_rul(small_fleet, params["prepare"]["rul_cap"])
    train = fleet[fleet["unit_id"].isin(PARTS["train"])]
    scaled = scale_sensors(train, fit_scaler(train, sensors), sensors)
    for folder, table in [
        ("processed", scaled),
        ("features", add_rolling_features(scaled, sensors, params["features"]["rolling_window"])),
    ]:
        (tmp_path / "data" / folder).mkdir(parents=True)
        table.to_parquet(tmp_path / "data" / folder / "train.parquet", index=False)
    fleet.to_parquet(tmp_path / "data" / "processed" / "test.parquet", index=False)

    reports = tmp_path / "reports"
    parts = {name: {"n_engines": len(units), "units": units} for name, units in PARTS.items()}
    write_json(reports / "prepare.json", {"parts": parts | {"test": {"n_engines": 8}}})
    write_json(
        reports / "champion.json", {"champion": "lstm", "registry_version": "1", "test_rmse": 15.0}
    )
    write_json(reports / "metrics" / "test.json", {name: scores(20.0) for name in MODELS})
    write_json(
        reports / "metrics" / "val_classic.json", {name: scores(18.0) for name in MODELS[:4]}
    )
    write_json(reports / "metrics" / "val_deep.json", {name: scores(12.0) for name in MODELS[4:]})
    pd.DataFrame(
        [
            {"model": model, "unit_id": unit, "rul_true": 10.0 * unit, "rul_pred": 10.0 * unit + 5}
            for model in MODELS
            for unit in range(1, 9)
        ]
    ).assign(
        rul_low=lambda df: df["rul_pred"] - 20, rul_high=lambda df: df["rul_pred"] + 20
    ).to_parquet(reports / "predictions_test.parquet", index=False)
    pd.DataFrame(
        [
            {"epoch": epoch, "train_loss": 1 / epoch, "val_rmse": 20 - epoch, "model": model}
            for model in ["lstm", "cnn"]
            for epoch in range(1, 7)
        ]
    ).to_csv(reports / "training_curves.csv", index=False)

    explain = reports / "explain"
    explain.mkdir()
    inputs = [f"{sensor}_mean" for sensor in sensors]
    pd.Series(np.linspace(5, 1, len(inputs)), index=inputs, name="mean_abs_shap").to_csv(
        explain / "xgboost_shap_importance.csv"
    )
    pd.Series(np.linspace(9, 1, len(sensors)), index=sensors, name="rmse_increase").to_csv(
        explain / "lstm_permutation_importance.csv"
    )
    ranks = np.arange(1, len(sensors) + 1)
    pd.DataFrame(
        {
            "sensor": sensors,
            "xgboost_shap": ranks[::-1] * 1.0,
            "lstm_permutation": ranks[::-1] * 2.0,
            "xgboost_rank": ranks,
            "lstm_rank": ranks,
        }
    ).to_csv(explain / "sensor_ranking.csv", index=False)
    write_json(
        explain / "one_prediction.json",
        {"unit_id": 5, "cycle": 30, "true_rul": 20, "base_value": 80.0, "prediction": 25.0},
    )
    for name in ("xgboost_shap_beeswarm.png", "xgboost_shap_one_prediction.png"):
        plt.imsave(explain / name, np.zeros((4, 4, 3)))  # a tiny black image

    write_simulation(tmp_path, small_fleet, s_11_shift=0.0)
    write_retraining_reports(tmp_path)
    return tmp_path


def write_retraining_reports(root) -> None:
    """Two runs of the retraining flow: one without drift, then one that promoted a challenger."""
    no_drift = {
        "started_at": "2026-11-22T10:00:00+00:00",
        "champion_version": "5",
        "drift_share": 0.07,
        "drifted_sensors": ["s_9"],
        "n_logged": 1374,
        "decision": {
            "retrain": False,
            "promote": False,
            "improvement": None,
            "fd001_change": None,
            "reason": "drift share 0.07 below 0.3: no retraining",
        },
    }
    promoted = no_drift | {
        "started_at": "2026-11-22T11:00:00+00:00",
        "drift_share": 0.93,
        "n_logged": 16596,
        "challenger_version": "6",
        "scores": {
            "champion": {"FD001": 15.0, "FD003": 40.0, "all": 30.0},
            "challenger": {"FD001": 15.5, "FD003": 24.0, "all": 20.0},
        },
        "decision": {
            "retrain": True,
            "promote": True,
            "improvement": 1 / 3,
            "fd001_change": 0.5 / 15,
            "reason": "RMSE on all test engines improved by 33.3% and FD001 changed by +3.3%",
        },
    }
    write_json(root / "reports" / "retraining" / "2026-11-22T100000Z.json", no_drift)
    write_json(root / "reports" / "retraining" / "2026-11-22T110000Z.json", promoted)


def write_simulation(root, fleet: pd.DataFrame, s_11_shift: float) -> None:
    """The two logs of a simulation: every cycle of the fleet predicted 10 cycles too high."""
    logged = fleet.assign(s_11=fleet["s_11"] + s_11_shift)
    truth = 200 - fleet["cycle"]  # capped at 125 by the page
    logged = logged.assign(rul=truth.clip(upper=125) + 10.0, rul_low=0.0, rul_high=125.0)
    monitoring = root / "monitoring"
    monitoring.mkdir(exist_ok=True)
    logged.to_json(monitoring / "predictions.jsonl", orient="records", lines=True)
    fleet[["unit_id", "cycle"]].assign(scenario="normal", rul_true=truth).to_json(
        monitoring / "ground_truth.jsonl", orient="records", lines=True
    )


@pytest.fixture
def answering_api(monkeypatch):
    """Replace the two calls to the API by answers made up from the engine's own cycles."""

    def model_info(url):
        return {"model_type": "lstm", "model_version": "1", "alert_threshold": 30}

    def prediction_curve(url, engine, model_version):
        cycles = engine["cycle"].sort_values().to_numpy()
        rul = np.clip(125 - cycles, 0, 125).astype(float)
        return pd.DataFrame(
            {"cycle": cycles, "rul": rul, "rul_low": rul - 10, "rul_high": rul + 10}
        ).assign(alert=lambda df: df["rul"] <= 30)

    monkeypatch.setattr(api_client, "model_info", model_info)
    monkeypatch.setattr(api_client, "prediction_curve", prediction_curve)


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30).run()
    return app.switch_page(page).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_without_the_pipeline_files_says_what_to_run(page, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(tmp_path))  # an empty folder
    app = open_page(page)
    assert not app.exception
    assert "Run `" in app.info[0].value


@pytest.mark.parametrize("page", PAGES)
def test_page_with_toy_files_draws_without_error(page, toy_root, answering_api, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(page)
    assert not app.exception
    assert not app.info  # no file reported missing
    assert not app.error
    assert app.title[0].value


def test_overview_counts_the_engines_and_the_alerts(toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    metrics = {metric.label: metric.value for metric in open_page(PAGES[0]).metric}
    assert metrics["Training engines"] == "8"
    # predicted 15, 25, 35...: two of the eight engines are at 30 cycles or less
    assert metrics["Test engines in alert"] == "2 / 8"


def test_models_page_follows_the_chosen_model(toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(PAGES[3])
    assert app.segmented_control[0].value == "lstm"  # the champion is shown first
    app.segmented_control[0].set_value("ridge").run()
    assert not app.exception


def test_engine_view_says_how_to_start_the_api_when_it_does_not_answer(toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    app = open_page("views/engine_view.py")
    assert not app.exception
    assert "uvicorn api.main:app" in app.error[0].value


def test_engine_view_shows_the_prediction_at_the_last_cycle(
    toy_root, answering_api, small_fleet, monkeypatch
):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    metrics = {m.label: m.value for m in open_page("views/engine_view.py").metric}
    # engine 34 is not among the toy engines, so the first one is shown
    n_cycles = int((small_fleet["unit_id"] == 1).sum())
    assert metrics["Cycles seen"] == str(n_cycles)
    # the made-up API predicts 125 - cycle; the toy engines have 40 to 80 cycles
    assert metrics["Predicted remaining life"] == f"{125 - n_cycles} cycles"
    assert metrics["Alert"] == "no"


def test_glossary_filters_the_terms(monkeypatch):
    app = AppTest.from_file(APP, default_timeout=30).run().switch_page(GLOSSARY).run()
    n_terms = len(app.markdown)
    app.text_input[0].set_value("leakage").run()
    assert not app.exception
    assert 0 < len(app.markdown) < n_terms
    app.text_input[0].set_value("no such word").run()
    assert "No term matches" in app.info[0].value


def test_live_monitoring_without_a_log_says_to_run_a_simulation(toy_root, monkeypatch):
    (toy_root / "monitoring" / "predictions.jsonl").unlink()
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(LIVE)
    assert not app.exception
    assert "python -m pipeline.simulate" in app.info[0].value


def test_live_monitoring_shows_the_error_and_no_drift_on_unchanged_sensors(toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    metrics = {m.label: m.value for m in open_page(LIVE).metric}
    assert metrics["Scenario"] == "normal"
    assert metrics["Engines"] == "8"
    # the reference is the same toy fleet: nothing drifts
    assert metrics["Drift share"] == "0.00"
    # every prediction is 10 cycles too high; the toy fleet has fewer than 2000 cycles in all
    assert metrics["RMSE, last 2000 predictions"] == "n/a"


def test_live_monitoring_puts_the_shifted_sensor_first(toy_root, small_fleet, monkeypatch):
    write_simulation(toy_root, small_fleet, s_11_shift=3.0)  # far more than s_11's spread
    params = (toy_root / "params.yaml").read_text(encoding="utf-8")
    params = params.replace("error_window: 2000", "error_window: 100")
    (toy_root / "params.yaml").write_text(params, encoding="utf-8")
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(LIVE)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Drift share"] == f"{1 / 14:.2f}"
    assert metrics["RMSE, last 100 predictions"] == "10.0 cycles"
    assert app.selectbox[0].value == "s_11"  # sensors sorted by distance
    assert not app.warning  # one sensor of 14 is below the retraining share


def test_retraining_shows_the_latest_run_and_every_run(toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(RETRAINING)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Drift share"] == "0.93"
    assert metrics["Challenger"] == "v6"
    assert metrics["Decision"] == "challenger promoted"
    assert "improved by 33.3%" in app.success[0].value
    history = app.dataframe[-1].value
    assert len(history) == 2
    assert history.iloc[0]["started_at"].startswith("2026-11-22T11")  # the latest first


def test_retraining_after_a_run_without_drift(toy_root, monkeypatch):
    (toy_root / "reports" / "retraining" / "2026-11-22T110000Z.json").unlink()
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(RETRAINING)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Challenger"] == "not trained"
    assert metrics["Decision"] == "no retraining"
    assert not app.exception


def test_overview_warns_when_retraining_replaced_the_champion(toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    assert not open_page(PAGES[0]).warning  # no bundle in the toy folder: nothing to say
    served = {"registry_version": "2", "model_type": "lstm", "subsets": ["FD001", "FD003"]}
    write_json(toy_root / "models" / "champion" / "metadata.json", served)
    warning = open_page(PAGES[0]).warning[0].value
    assert "promoted version 2 (LSTM, trained on FD001 + FD003)" in warning


def test_models_page_compares_the_subsets(toy_root, monkeypatch):
    def run(name, subset, by_regime, lstm_rmse):
        models = ["baseline", "ridge", "random_forest", "xgboost", "lstm", "cnn"]
        scores = {"val_rmse": 20.0, "test_rmse": 25.0, "test_mae": 18.0}
        scores |= {"test_nasa_score": 900.0, "coverage": 0.9}
        return {
            "name": name,
            "subset": subset,
            "normalize_by_regime": by_regime,
            "n_train_engines": 70,
            "n_test_engines": 100,
            "models": {
                m: scores | ({"test_rmse": lstm_rmse} if m == "lstm" else {}) for m in models
            },
        }

    runs = [run("pipeline", "FD001", False, 15.0), run("fd002-regimes", "FD002", True, 28.0)]
    write_json(toy_root / "reports" / "subsets.json", runs)
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(PAGES[3])
    assert not app.exception
    chooser = app.segmented_control[0]
    assert chooser.value == "pipeline"
    chooser.set_value("fd002-regimes").run()
    table = app.dataframe[0].value
    assert table.set_index("model").loc["lstm", "test_rmse"] == 28.0
