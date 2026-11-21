import pytest

from turbofan.config import load_params
from turbofan.data import load_raw
from turbofan.describe import engine_lifetimes, sensor_summary


def test_lifetime_is_the_last_cycle_of_each_engine(small_fleet):
    lifetimes = engine_lifetimes(small_fleet)
    assert len(lifetimes) == 8
    for unit_id, engine in small_fleet.groupby("unit_id"):
        assert lifetimes[unit_id] == len(engine)


def test_summary_explains_why_a_constant_sensor_is_dropped(small_fleet):
    summary = sensor_summary(small_fleet, kept=["s_2", "s_11"])
    assert len(summary) == 21
    assert summary.loc["s_1", "distinct_values"] == 1  # s_1 is 100.0 everywhere
    assert summary.loc["s_1", "reason"] == "dropped: constant, a single value"
    assert summary.loc["s_2", "kept"]
    assert summary.loc["s_2", "reason"] == "used by the models"
    assert summary.loc["s_3", "reason"] == "dropped"  # varies, but not kept here


@pytest.mark.needs_data
def test_on_fd001_the_seven_dropped_sensors_have_at_most_two_values():
    summary = sensor_summary(load_raw("FD001", "train"), load_params()["prepare"]["sensors"])
    dropped = summary[~summary["kept"]]
    assert dropped.index.tolist() == ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
    assert (dropped["distinct_values"] <= 2).all()
    assert engine_lifetimes(load_raw("FD001", "train")).agg(["min", "median", "max"]).tolist() == [
        128,
        199,
        362,
    ]
