"""Split engines into train, validation and calibration sets, and scale the sensors."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_units(unit_ids: list[int], n_val: int, n_calib: int, seed: int) -> dict[str, list[int]]:
    """Randomly assign whole engines to train, val and calib. Each list is sorted."""
    # split by engine, never by row: rows of one engine are almost identical from one cycle
    # to the next, so a row split would put the same engine in train and validation
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(unit_ids)).tolist()
    return {
        "train": sorted(shuffled[n_val + n_calib :]),
        "val": sorted(shuffled[:n_val]),
        "calib": sorted(shuffled[n_val : n_val + n_calib]),
    }


def fit_scaler(train_df: pd.DataFrame, sensors: list[str]) -> StandardScaler:
    """Learn the mean and standard deviation of each sensor on the training engines."""
    # fit on training engines only, otherwise validation scores are too optimistic
    return StandardScaler().fit(train_df[sensors])


def scale_sensors(df: pd.DataFrame, scaler: StandardScaler, sensors: list[str]) -> pd.DataFrame:
    """Return a copy of df where each sensor is replaced by (value - train mean) / train std."""
    df = df.copy()
    # replace the whole columns: writing into them with .loc fails on integer sensors such as s_17
    df[sensors] = scaler.transform(df[sensors])
    return df
