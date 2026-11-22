import json

import numpy as np
import pandas as pd
import pytest

from pipeline.monitor import load_reference
from turbofan.config import load_params
from turbofan.data import load_raw
from turbofan.monitoring import build_drift_report, load_prediction_log


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
