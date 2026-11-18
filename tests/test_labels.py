import pandas as pd
import pytest

from turbofan.data import load_raw
from turbofan.labels import add_rul, add_test_rul, clip_rul


def make_engines(lengths: dict[int, int]) -> pd.DataFrame:
    """Return a table with only unit_id and cycle, one row per cycle of each engine."""
    rows = [(unit_id, cycle) for unit_id, n in lengths.items() for cycle in range(1, n + 1)]
    return pd.DataFrame(rows, columns=["unit_id", "cycle"])


def test_rul_counts_down_to_zero_at_last_cycle():
    df = add_rul(make_engines({1: 5}), cap=125)

    assert df["rul"].tolist() == [4, 3, 2, 1, 0]


def test_rul_clipped_at_cap():
    df = add_rul(make_engines({1: 5}), cap=3)

    assert df["rul_clipped"].tolist() == [3, 3, 2, 1, 0]
    assert df["rul"].tolist() == [4, 3, 2, 1, 0]


def test_each_engine_ends_at_zero():
    df = add_rul(make_engines({1: 3, 2: 5}), cap=125)

    assert df.loc[df["unit_id"] == 1, "rul"].tolist() == [2, 1, 0]
    assert df.loc[df["unit_id"] == 2, "rul"].tolist() == [4, 3, 2, 1, 0]


def test_add_rul_leaves_input_unchanged():
    engines = make_engines({1: 3})

    add_rul(engines, cap=125)

    assert list(engines.columns) == ["unit_id", "cycle"]


def test_clip_rul_keeps_values_below_cap():
    assert clip_rul(pd.Series([200, 125, 124, 0]), cap=125).tolist() == [125, 125, 124, 0]


def test_test_rul_counts_down_to_final_rul():
    final_rul = pd.Series([10, 2], index=pd.Index([1, 2], name="unit_id"))

    df = add_test_rul(make_engines({1: 3, 2: 2}), final_rul, cap=11)

    # engine 1: 10 cycles left at its last row (cycle 3), so 12 at cycle 1
    assert df.loc[df["unit_id"] == 1, "rul"].tolist() == [12, 11, 10]
    assert df.loc[df["unit_id"] == 1, "rul_clipped"].tolist() == [11, 11, 10]
    assert df.loc[df["unit_id"] == 1, "final_rul"].tolist() == [10, 10, 10]
    assert df.loc[df["unit_id"] == 2, "rul"].tolist() == [3, 2]


# reference values computed on the real file on 13/11/2026 (planning notes)


@pytest.mark.needs_data
def test_real_fd001_train_rul():
    df = add_rul(load_raw("FD001", "train"), cap=125)

    assert df["rul"].sum() == 2224184
    assert (df["rul_clipped"] == 125).sum() == 8131
    first_row = df.iloc[0]
    assert first_row["rul"] == 191
    assert first_row["rul_clipped"] == 125
