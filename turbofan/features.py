"""Rolling features for the classic models, which only see one row at a time."""

import pandas as pd


def add_rolling_features(df: pd.DataFrame, sensors: list[str], window: int) -> pd.DataFrame:
    """Add {sensor}_mean and {sensor}_std over the last `window` cycles of the same engine.

    Each value only uses the current and earlier cycles of its own engine.
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    # rolling inside each engine: a window never mixes the end of one engine with the next one
    rolling = df.groupby("unit_id")[sensors].rolling(window, min_periods=1)
    # the result is indexed by (unit_id, row); dropping unit_id aligns it with df again
    means = rolling.mean().reset_index(level=0, drop=True)
    # the standard deviation of a single value is undefined (NaN): the spread is 0
    stds = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
    for sensor in sensors:
        df[f"{sensor}_mean"] = means[sensor]
        df[f"{sensor}_std"] = stds[sensor]
    return df


def feature_columns(sensors: list[str]) -> list[str]:
    """Return the model inputs: for each sensor, its value, rolling mean and rolling std."""
    columns = []
    for sensor in sensors:
        columns += [sensor, f"{sensor}_mean", f"{sensor}_std"]
    return columns


def last_row_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Return the row with the highest cycle of each engine, sorted by unit_id."""
    last_rows = df.loc[df.groupby("unit_id")["cycle"].idxmax()]
    return last_rows.sort_values("unit_id").reset_index(drop=True)
