import pandas as pd
import pytest

from turbofan.features import add_rolling_features, feature_columns, last_row_per_unit


def make_engines(values_by_unit: dict[int, list[float]]) -> pd.DataFrame:
    """One sensor s_a, one row per cycle, cycles numbered from 1 inside each engine."""
    rows = [
        (unit_id, cycle, value)
        for unit_id, values in values_by_unit.items()
        for cycle, value in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows, columns=["unit_id", "cycle", "s_a"])


def test_rolling_mean_and_std_by_hand():
    df = add_rolling_features(make_engines({1: [1.0, 2.0, 3.0, 4.0]}), ["s_a"], window=2)

    assert df["s_a_mean"].tolist() == [1.0, 1.5, 2.5, 3.5]
    # sample standard deviation of two consecutive integers is sqrt(0.5) = 0.7071
    assert df["s_a_std"].tolist() == pytest.approx([0.0, 0.7071, 0.7071, 0.7071], abs=1e-4)


def test_second_engine_starts_fresh():
    df = add_rolling_features(make_engines({1: [1.0, 2.0, 3.0], 2: [100.0, 200.0]}), ["s_a"], 3)
    second = df[df["unit_id"] == 2]

    # nothing from engine 1 leaks into the first row of engine 2
    assert second["s_a_mean"].tolist() == [100.0, 150.0]
    assert second["s_a_std"].iloc[0] == 0.0


def test_no_look_ahead():
    before = add_rolling_features(make_engines({1: [1.0, 2.0, 3.0, 4.0]}), ["s_a"], window=3)
    changed = make_engines({1: [1.0, 2.0, 3.0, 400.0]})
    after = add_rolling_features(changed, ["s_a"], window=3)

    # changing the last cycle cannot change any feature of the earlier cycles
    pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
    assert after["s_a_mean"].iloc[3] != before["s_a_mean"].iloc[3]


def test_rows_are_put_back_in_cycle_order():
    shuffled = make_engines({1: [1.0, 2.0, 3.0]}).iloc[[2, 0, 1]]

    df = add_rolling_features(shuffled, ["s_a"], window=2)

    assert df["cycle"].tolist() == [1, 2, 3]
    assert df["s_a_mean"].tolist() == [1.0, 1.5, 2.5]


def test_feature_columns_are_three_per_sensor_in_stable_order():
    assert feature_columns(["s_2", "s_3"]) == [
        "s_2",
        "s_2_mean",
        "s_2_std",
        "s_3",
        "s_3_mean",
        "s_3_std",
    ]


def test_last_row_per_unit_takes_the_highest_cycle():
    df = make_engines({2: [5.0, 6.0], 1: [1.0, 2.0, 3.0]})

    last_rows = last_row_per_unit(df)

    assert last_rows["unit_id"].tolist() == [1, 2]
    assert last_rows["cycle"].tolist() == [3, 2]
    assert last_rows["s_a"].tolist() == [3.0, 6.0]
