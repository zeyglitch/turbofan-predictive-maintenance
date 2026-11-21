"""Descriptive figures on the raw data, shown by the dashboard: no model, no fitting."""

import pandas as pd

from turbofan.config import SENSOR_COLUMNS


def engine_lifetimes(df: pd.DataFrame) -> pd.Series:
    """Number of cycles of each engine (its last cycle), indexed by unit_id."""
    return df.groupby("unit_id")["cycle"].max().rename("lifetime")


def sensor_summary(df: pd.DataFrame, kept: list[str]) -> pd.DataFrame:
    """One row per raw sensor: spread, number of distinct values, used or not, and why.

    The rule of phase 1: a sensor with at most two distinct values says nothing about wear.
    """
    summary = pd.DataFrame(
        {"std": df[SENSOR_COLUMNS].std(), "distinct_values": df[SENSOR_COLUMNS].nunique()}
    )
    summary["kept"] = summary.index.isin(kept)
    pairs = zip(summary["kept"], summary["distinct_values"], strict=True)
    summary["reason"] = [reason(is_kept, n) for is_kept, n in pairs]
    return summary.rename_axis("sensor")


def reason(is_kept: bool, distinct_values: int) -> str:
    if is_kept:
        return "used by the models"
    if distinct_values == 1:
        return "dropped: constant, a single value"
    if distinct_values == 2:
        return "dropped: only two values"
    return "dropped"
