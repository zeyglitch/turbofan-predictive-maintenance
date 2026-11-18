import numpy as np
import pandas as pd
import pytest

from turbofan.preprocessing import fit_scaler, scale_sensors, split_units

UNIT_IDS = list(range(1, 101))


def test_split_sizes_are_70_15_15():
    parts = split_units(UNIT_IDS, n_val=15, n_calib=15, seed=42)

    assert len(parts["train"]) == 70
    assert len(parts["val"]) == 15
    assert len(parts["calib"]) == 15


def test_split_parts_are_disjoint_and_cover_all_engines():
    parts = split_units(UNIT_IDS, n_val=15, n_calib=15, seed=42)
    train, val, calib = set(parts["train"]), set(parts["val"]), set(parts["calib"])

    assert train & val == set()
    assert train & calib == set()
    assert val & calib == set()
    assert train | val | calib == set(UNIT_IDS)


def test_same_seed_gives_same_split_and_other_seed_differs():
    first = split_units(UNIT_IDS, n_val=15, n_calib=15, seed=42)
    second = split_units(UNIT_IDS, n_val=15, n_calib=15, seed=42)
    other = split_units(UNIT_IDS, n_val=15, n_calib=15, seed=7)

    assert first == second
    assert first != other


def make_sensor_table(values_a: list[float], values_b: list[float]) -> pd.DataFrame:
    n = len(values_a)
    return pd.DataFrame(
        {
            "unit_id": [1] * n,
            "cycle": list(range(1, n + 1)),
            "s_a": values_a,
            "s_b": values_b,
            "rul": list(range(n - 1, -1, -1)),
        }
    )


def test_scaled_train_sensors_have_mean_0_and_std_1():
    train = make_sensor_table([1.0, 2.0, 3.0, 4.0], [10.0, 30.0, 20.0, 40.0])
    scaler = fit_scaler(train, ["s_a", "s_b"])

    scaled = scale_sensors(train, scaler, ["s_a", "s_b"])

    # StandardScaler divides by the population standard deviation (ddof=0)
    assert scaled[["s_a", "s_b"]].mean().tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert scaled[["s_a", "s_b"]].std(ddof=0).tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_other_data_is_scaled_with_train_statistics():
    train = make_sensor_table([1.0, 2.0, 3.0, 4.0], [10.0, 30.0, 20.0, 40.0])
    val = make_sensor_table([5.0, 6.0], [50.0, 60.0])
    scaler = fit_scaler(train, ["s_a", "s_b"])

    scaled_val = scale_sensors(val, scaler, ["s_a", "s_b"])

    # s_a on train: mean 2.5, population std sqrt(1.25), so 5 becomes (5 - 2.5) / sqrt(1.25)
    assert scaled_val.loc[0, "s_a"] == pytest.approx((5.0 - 2.5) / np.sqrt(1.25))
    # validation is not centred: its own mean is far from 0
    assert scaled_val["s_a"].mean() > 2


def test_integer_sensor_is_scaled_to_floats():
    # s_17 is stored as integers in the raw files
    train = make_sensor_table([1.0, 2.0, 3.0, 4.0], [10.0, 30.0, 20.0, 40.0])
    train["s_b"] = train["s_b"].astype("int64")
    scaler = fit_scaler(train, ["s_a", "s_b"])

    scaled = scale_sensors(train, scaler, ["s_a", "s_b"])

    assert scaled["s_b"].dtype == "float64"
    assert scaled["s_b"].mean() == pytest.approx(0.0, abs=1e-6)


def test_scaling_leaves_other_columns_and_input_unchanged():
    train = make_sensor_table([1.0, 2.0, 3.0, 4.0], [10.0, 30.0, 20.0, 40.0])
    original = train.copy()
    scaler = fit_scaler(train, ["s_a", "s_b"])

    scaled = scale_sensors(train, scaler, ["s_a", "s_b"])

    for column in ["unit_id", "cycle", "rul"]:
        assert scaled[column].tolist() == original[column].tolist()
    pd.testing.assert_frame_equal(train, original)
