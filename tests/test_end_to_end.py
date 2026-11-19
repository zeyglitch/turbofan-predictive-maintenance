import numpy as np

from turbofan.classic_models import train_baseline, train_ridge
from turbofan.features import add_rolling_features, feature_columns
from turbofan.labels import add_rul
from turbofan.metrics import regression_report
from turbofan.preprocessing import fit_scaler, scale_sensors, split_units


def test_chain_runs_on_a_small_fleet_and_ridge_beats_the_baseline(small_fleet):
    """Target, engine split, scaling, rolling features, Ridge and metrics, chained on toy data."""
    sensors = ["s_2", "s_3", "s_11"]

    df = add_rul(small_fleet, cap=50)
    units = split_units(df["unit_id"].unique().tolist(), n_val=2, n_calib=2, seed=0)
    parts = {name: df[df["unit_id"].isin(ids)] for name, ids in units.items()}
    scaler = fit_scaler(parts["train"], sensors)
    parts = {
        name: add_rolling_features(scale_sensors(part, scaler, sensors), sensors, window=5)
        for name, part in parts.items()
    }
    train, val = parts["train"], parts["val"]

    ridge = train_ridge(train[feature_columns(sensors)], train["rul_clipped"], {"alpha": 1.0})
    baseline = train_baseline(train["rul_clipped"])
    ridge_scores = regression_report(
        val["rul_clipped"], ridge.predict(val[feature_columns(sensors)])
    )
    baseline_scores = regression_report(val["rul_clipped"], np.full(len(val), baseline))

    assert [len(units[name]) for name in ["train", "val", "calib"]] == [4, 2, 2]
    assert ridge_scores["rmse"] < baseline_scores["rmse"]
