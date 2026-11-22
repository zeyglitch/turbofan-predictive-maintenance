import json

import numpy as np
import pandas as pd
import pytest

from pipeline.monitor import error_summary, load_reference
from turbofan.config import load_params
from turbofan.data import load_raw
from turbofan.monitoring import (
    build_drift_report,
    error_over_time,
    join_ground_truth,
    load_prediction_log,
    reference_period,
)


def two_samples(shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference and current with the same noise; s_b of current moved by shift reference stds."""
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"s_a": rng.normal(10, 2, 3000), "s_b": rng.normal(0, 0.5, 3000)})
    current = pd.DataFrame({"s_a": rng.normal(10, 2, 2000), "s_b": rng.normal(0, 0.5, 2000)})
    current["s_b"] += shift * 0.5
    return reference, current


def test_same_distribution_does_not_drift():
    reference, current = two_samples(shift=0.0)
    report = build_drift_report(reference, current, ["s_a", "s_b"], threshold=0.1)
    assert report["drift_share"] == 0
    assert report["drifted_sensors"] == []


def test_a_shifted_sensor_drifts_and_its_distance_is_in_reference_stds():
    reference, current = two_samples(shift=0.5)
    report = build_drift_report(reference, current, ["s_a", "s_b"], threshold=0.1)
    assert report["drifted_sensors"] == ["s_b"]
    assert report["drift_share"] == 0.5
    rows = {row["sensor"]: row for row in report["sensors"]}
    # a pure shift of the whole distribution: the distance is the shift itself
    assert rows["s_b"]["distance"] == pytest.approx(0.5, abs=0.05)
    assert rows["s_a"]["distance"] < 0.1
    # with thousands of rows, the test p-value says "different" long before it matters
    assert rows["s_b"]["ks_pvalue"] < 1e-6


def test_report_can_be_written_as_json():
    reference, current = two_samples(shift=0.5)
    json.dumps(build_drift_report(reference, current, ["s_a", "s_b"], threshold=0.1))


def test_log_is_read_in_order_of_arrival(tmp_path):
    path = tmp_path / "predictions.jsonl"
    lines = [{"unit_id": 7, "cycle": c, "s_2": 640.0 + c, "rul": 100.0 - c} for c in (1, 2, 3)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    log = load_prediction_log(path)
    assert log["cycle"].tolist() == [1, 2, 3]
    assert log.loc[2, "s_2"] == 643.0


def test_reference_period_keeps_the_first_engines():
    test = pd.DataFrame({"unit_id": [1, 1, 2, 3, 3], "cycle": [1, 2, 1, 1, 2]})
    assert reference_period(test, 2)["unit_id"].tolist() == [1, 1, 2]


def test_truth_is_joined_in_order_of_arrival_and_unknown_predictions_are_left_out():
    log = pd.DataFrame({"unit_id": [51, 51, 34, 52], "cycle": [2, 1, 9, 1], "rul": [1.0] * 4})
    truth = pd.DataFrame(
        {"unit_id": [51, 51, 52, 51], "cycle": [1, 2, 1, 2], "rul_true": [130, 129, 80, 999]}
    )
    joined = join_ground_truth(log, truth)
    # engine 34 has no truth (replayed by the dashboard, say); the order of the log is kept
    assert joined[["unit_id", "cycle"]].values.tolist() == [[51, 2], [51, 1], [52, 1]]
    # the same cycle written twice: the last truth wins
    assert joined["rul_true"].tolist() == [999, 130, 80]


def test_error_uses_the_capped_truth_and_the_rolling_rmse_waits_for_a_full_window():
    joined = pd.DataFrame({"rul": [120.0, 50.0, 40.0, 30.0], "rul_true": [200, 53, 36, 30]})
    errors = error_over_time(joined, cap=125, window=2)
    assert errors["rul_true_clipped"].tolist() == [125, 53, 36, 30]
    assert errors["error"].tolist() == [-5.0, -3.0, 4.0, 0.0]
    assert np.isnan(errors.loc[0, "rolling_rmse"])  # one prediction is not a window of two
    assert errors.loc[1, "rolling_rmse"] == pytest.approx(np.sqrt((25 + 9) / 2))
    assert errors.loc[3, "rolling_rmse"] == pytest.approx(np.sqrt(16 / 2))


def test_error_summary_splits_the_replay_in_two_halves():
    log = pd.DataFrame({"unit_id": 1, "cycle": [1, 2, 3, 4], "rul": [10.0, 10.0, 10.0, 20.0]})
    truth = pd.DataFrame({"unit_id": 1, "cycle": [1, 2, 3, 4], "rul_true": [10, 10, 10, 10]})
    summary = error_summary(log, truth, cap=125, settings={"error_window": 10})
    assert summary["n_with_truth"] == 4
    assert summary["rmse_first_half"] == 0
    assert summary["rmse_second_half"] == pytest.approx(np.sqrt(100 / 2))
    assert summary["last_window_rmse"] is None  # 4 predictions, a window of 10
    json.dumps(summary)


@pytest.mark.needs_data
def test_two_halves_of_the_fd001_test_engines_do_not_drift():
    params = load_params()
    settings = params["monitoring"]
    reference = load_reference(settings)
    assert reference["unit_id"].max() == settings["reference_engines"]
    test = load_raw("FD001", "test")
    others = test[test["unit_id"] > settings["reference_engines"]]
    report = build_drift_report(
        reference, others, params["prepare"]["sensors"], settings["drift_threshold"]
    )
    assert report["drift_share"] == 0
