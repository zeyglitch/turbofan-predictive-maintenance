import pandas as pd
import pytest

from turbofan.config import load_params
from turbofan.data import load_raw, load_test_rul
from turbofan.simulate import apply_sensor_offset, iter_engine_cycles, true_rul


def test_every_engine_is_replayed_from_its_first_cycle(small_fleet):
    histories = list(iter_engine_cycles(small_fleet))
    assert len(histories) == len(small_fleet)  # one request per row
    unit_id, first = histories[0]
    assert unit_id == 1
    assert first["cycle"].tolist() == [1]
    _, second = histories[1]
    assert second["cycle"].tolist() == [1, 2]  # the history grows by one cycle
    engine_8 = small_fleet[small_fleet["unit_id"] == 8]
    unit_id, last = histories[-1]
    assert unit_id == 8
    assert len(last) == len(engine_8)  # the last request carries the whole life


def test_with_a_step_the_last_cycle_is_still_sent(small_fleet):
    engine = small_fleet[small_fleet["unit_id"] == 1]
    lengths = [len(history) for _, history in iter_engine_cycles(engine, step=10)]
    assert lengths[-1] == len(engine)
    assert all(b - a == 10 for a, b in zip(lengths, lengths[1:], strict=False))
    assert lengths[0] <= 10


def test_true_rul_counts_down_to_the_value_of_the_rul_file():
    df = pd.DataFrame({"unit_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    test_rul = pd.Series([10, 50], index=pd.Index([1, 2], name="unit_id"))
    assert true_rul(df, test_rul).tolist() == [12, 11, 10, 51, 50]


def test_offset_starts_at_the_given_fraction_and_reaches_its_maximum_on_the_last_row(small_fleet):
    sensor_std = pd.Series({"s_2": 2.0, "s_11": 0.5})
    shifted = apply_sensor_offset(small_fleet, ["s_2", "s_11"], sensor_std, 1.5, 0.5)
    original = small_fleet.sort_values(["unit_id", "cycle"])
    offset = shifted["s_2"].to_numpy() - original["s_2"].to_numpy()
    half = len(offset) // 2
    assert (offset[:half] == 0).all()  # first half of the replay untouched
    assert offset[-1] == pytest.approx(1.5 * 2.0)
    assert (offset[half:][1:] >= offset[half:][:-1]).all()  # then it only grows
    assert shifted["s_11"].iloc[-1] - original["s_11"].iloc[-1] == pytest.approx(1.5 * 0.5)
    assert shifted["s_3"].equals(original["s_3"])  # other sensors unchanged
    assert small_fleet["s_2"].equals(original["s_2"].sort_index())  # input not modified


@pytest.mark.needs_data
def test_on_fd001_the_true_rul_of_the_last_cycles_is_the_rul_file():
    test = load_raw("FD001", "test")
    rul = true_rul(test, load_test_rul("FD001"))
    last_rows = test.groupby("unit_id")["cycle"].idxmax()
    assert rul[last_rows].tolist() == load_test_rul("FD001").tolist()
    assert rul.min() == 7  # engine 34, the closest to failure of the test set
    assert load_params()["monitoring"]["offset_sensors"] == ["s_11", "s_4"]
