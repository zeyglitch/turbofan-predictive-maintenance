"""Sliding windows of consecutive cycles, the input of the neural networks."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def pad_start(values: np.ndarray, length: int) -> np.ndarray:
    """Repeat the first row at the start until there are at least `length` rows."""
    missing = length - len(values)
    if missing <= 0:
        return values
    return np.concatenate([np.repeat(values[:1], missing, axis=0), values])


def make_windows(
    df: pd.DataFrame, sensors: list[str], length: int, every_cycle: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut each engine into every window of `length` consecutive cycles.

    With every_cycle=True, one window ends on each cycle, including the first length - 1 ones
    (padded at the start), so that the windows match the rows of df one to one.
    Returns X of shape (n, length, n_sensors), the rul_clipped of the last cycle of each window
    (NaN when df has no rul_clipped column) and the unit_id of each window.
    """
    df = df.sort_values(["unit_id", "cycle"])
    X_parts, y_parts, unit_parts = [], [], []
    # one engine at a time: a window never contains the end of one engine and the start of the next
    for unit_id, engine in df.groupby("unit_id", sort=True):
        values = engine[sensors].to_numpy(np.float32)
        # length - 1 extra rows in front give a full window ending on the very first cycle
        values = pad_start(values, len(values) + length - 1 if every_cycle else length)
        # (T, sensors) -> (T - length + 1, sensors, length), then swap to (…, length, sensors)
        windows = sliding_window_view(values, length, axis=0).transpose(0, 2, 1)
        n = len(windows)
        if "rul_clipped" in engine:
            # window i ends on row i + length - 1: its target is the rul of that row
            y_parts.append(engine["rul_clipped"].to_numpy(np.float32)[-n:])
        else:
            y_parts.append(np.full(n, np.nan, dtype=np.float32))
        X_parts.append(windows)
        unit_parts.append(np.full(n, unit_id))
    return np.concatenate(X_parts), np.concatenate(y_parts), np.concatenate(unit_parts)


def last_window_per_unit(
    df: pd.DataFrame, sensors: list[str], length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the window ending on the last cycle of each engine, and the unit_ids, sorted."""
    last_rows = df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(length)
    # at most `length` rows per engine: make_windows cuts exactly one window from each
    X, _, unit_ids = make_windows(last_rows, sensors, length)
    return X, unit_ids
