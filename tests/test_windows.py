import numpy as np
import pandas as pd
import pytest

from turbofan.config import load_params
from turbofan.data import load_raw
from turbofan.windows import last_window_per_unit, make_windows

SENSORS = ["s_a", "s_b"]


def engine(unit_id: int, n_cycles: int) -> pd.DataFrame:
    """One engine whose sensor values encode the engine and the cycle: s_a = 1000 * unit + cycle."""
    cycle = np.arange(1, n_cycles + 1)
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "cycle": cycle,
            "s_a": 1000.0 * unit_id + cycle,
            "s_b": -cycle.astype(float),
            "rul_clipped": (n_cycles - cycle).astype(float),
        }
    )


def test_35_cycles_give_6_windows_and_the_last_one_ends_on_cycle_35():
    X, y, unit_ids = make_windows(engine(1, 35), SENSORS, 30)
    assert len(X) == len(y) == len(unit_ids) == 6
    np.testing.assert_array_equal(X[-1, :, 0], 1000 + np.arange(6, 36))
    assert y[-1] == 0  # rul_clipped of cycle 35, the last one
    assert y[0] == 5  # first window ends on cycle 30: 35 - 30 = 5 cycles left


def test_no_window_mixes_two_engines():
    df = pd.concat([engine(1, 35), engine(2, 40)], ignore_index=True)
    X, _, unit_ids = make_windows(df, SENSORS, 30)
    assert len(X) == 6 + 11
    # s_a // 1000 is the engine number of each row: one value per window
    engine_of_rows = X[:, :, 0] // 1000
    assert (engine_of_rows.min(axis=1) == engine_of_rows.max(axis=1)).all()
    np.testing.assert_array_equal(engine_of_rows[:, 0], unit_ids)


def test_shape_and_type():
    X, y, _ = make_windows(engine(1, 35), SENSORS, 30)
    assert X.shape == (6, 30, 2)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_short_engine_is_padded_at_the_start_with_its_first_row():
    X, y, _ = make_windows(engine(1, 10), SENSORS, 30)
    assert X.shape == (1, 30, 2)
    np.testing.assert_array_equal(X[0, :20], np.tile(X[0, 0], (20, 1)))
    np.testing.assert_array_equal(X[0, 20:, 0], 1000 + np.arange(1, 11))
    assert y[0] == 0


def test_every_cycle_gives_one_window_per_row():
    X, y, _ = make_windows(engine(1, 35), SENSORS, 30, every_cycle=True)
    assert X.shape == (35, 30, 2)
    np.testing.assert_array_equal(y, 35 - np.arange(1, 36))  # the rul of cycles 1 to 35
    np.testing.assert_array_equal(X[0, :, 0], np.full(30, 1001))  # cycle 1, repeated 30 times
    np.testing.assert_array_equal(X[9, -10:, 0], 1000 + np.arange(1, 11))  # ends on cycle 10
    np.testing.assert_array_equal(X[9, :20, 0], np.full(20, 1001))
    np.testing.assert_array_equal(X[-1, :, 0], 1000 + np.arange(6, 36))  # as without padding


def test_target_is_nan_without_rul_column():
    _, y, _ = make_windows(engine(1, 35).drop(columns="rul_clipped"), SENSORS, 30)
    assert np.isnan(y).all()


def test_rows_out_of_order_are_sorted_by_cycle_first():
    shuffled = engine(1, 35).sample(frac=1, random_state=0)
    X, _, _ = make_windows(shuffled, SENSORS, 30)
    np.testing.assert_array_equal(X[-1, :, 0], 1000 + np.arange(6, 36))


def test_last_window_per_unit_gives_one_window_per_engine_in_unit_order():
    df = pd.concat([engine(3, 40), engine(1, 35), engine(2, 10)], ignore_index=True)
    X, unit_ids = last_window_per_unit(df, SENSORS, 30)
    assert X.shape == (3, 30, 2)
    np.testing.assert_array_equal(unit_ids, [1, 2, 3])
    np.testing.assert_array_equal(X[0, :, 0], 1000 + np.arange(6, 36))  # engine 1, cycles 6-35
    assert X[1, -1, 0] == 2010  # engine 2 ends on its cycle 10
    assert X[2, -1, 0] == 3040  # engine 3 ends on its cycle 40


@pytest.mark.needs_data
def test_real_fd001_train_gives_17731_windows():
    sensors = load_params()["prepare"]["sensors"]
    X, _, _ = make_windows(load_raw("FD001", "train"), sensors, 30)
    assert X.shape == (17_731, 30, 14)
