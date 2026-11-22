"""Simulated production: test engines replayed cycle by cycle, as if they were flying.

No HTTP here: these functions only prepare the data that pipeline/simulate.py sends to the API.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd


def iter_engine_cycles(df: pd.DataFrame, step: int = 1) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (unit_id, history up to cycle t) for each engine, one engine after the other.

    The whole life is replayed, from the first cycle: a production model sees engines at every
    stage, not only near failure. With step > 1, one cycle in step is sent, counted back from
    the last one, so the last cycle of every engine is always sent.
    """
    for unit_id, engine in df.groupby("unit_id", sort=True):
        engine = engine.sort_values("cycle")
        for n in range(len(engine), 0, -step)[::-1]:
            yield int(unit_id), engine.iloc[:n]


def true_rul(df: pd.DataFrame, test_rul: pd.Series) -> pd.Series:
    """True remaining life at every row of a test file.

    RUL_FDxxx.txt gives it at the last cycle of each engine; one cycle earlier, it is one more.
    """
    last_cycle = df.groupby("unit_id")["cycle"].transform("max")
    return df["unit_id"].map(test_rul) + last_cycle - df["cycle"]


def apply_sensor_offset(
    df: pd.DataFrame,
    sensors: list[str],
    sensor_std: pd.Series,
    max_std: float,
    start_fraction: float,
) -> pd.DataFrame:
    """Add a slowly growing offset to some sensors, as a drifting or badly calibrated sensor would.

    Rows are taken in replay order (engine after engine, cycle after cycle). Nothing changes
    before start_fraction of the replay; then the offset grows linearly, up to max_std standard
    deviations (sensor_std, one value per sensor) on the last row.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    position = np.linspace(0, 1, len(df))  # 0 on the first row of the replay, 1 on the last
    ramp = np.clip((position - start_fraction) / (1 - start_fraction), 0, 1)
    for sensor in sensors:
        df[sensor] = df[sensor] + ramp * max_std * sensor_std[sensor]
    return df
