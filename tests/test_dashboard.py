"""Each page of the dashboard loads without an exception, with or without the pipeline's files.

AppTest runs the app in the test process, without a browser or a server.
"""

import json
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from turbofan.config import COLUMNS, ROOT, load_params
from turbofan.features import add_rolling_features
from turbofan.labels import add_rul
from turbofan.preprocessing import fit_scaler, scale_sensors

APP = str(ROOT / "dashboard" / "app.py")
PAGES = ["views/overview.py", "views/data_explorer.py", "views/preprocessing.py", "views/models.py"]
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
    """A project folder holding small versions of every file the four pages read."""
    shutil.copy(ROOT / "params.yaml", tmp_path / "params.yaml")
    shutil.copy(ROOT / "dvc.yaml", tmp_path / "dvc.yaml")
    params = load_params(tmp_path / "params.yaml")
    sensors = params["prepare"]["sensors"]

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    small_fleet[COLUMNS].to_csv(raw_dir / "train_FD001.txt", sep=" ", header=False, index=False)

    fleet = add_rul(small_fleet, params["prepare"]["rul_cap"])
    train = fleet[fleet["unit_id"].isin(PARTS["train"])]
    scaled = scale_sensors(train, fit_scaler(train, sensors), sensors)
    for folder, table in [
        ("processed", scaled),
        ("features", add_rolling_features(scaled, sensors, params["features"]["rolling_window"])),
    ]:
        (tmp_path / "data" / folder).mkdir(parents=True)
        table.to_parquet(tmp_path / "data" / folder / "train.parquet", index=False)

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
    return tmp_path


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
def test_page_with_toy_files_draws_without_error(page, toy_root, monkeypatch):
    monkeypatch.setenv("DASHBOARD_ROOT", str(toy_root))
    app = open_page(page)
    assert not app.exception
    assert not app.info  # no file reported missing
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
