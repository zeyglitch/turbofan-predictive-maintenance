"""Flight regimes (FD002, FD004): each row flies at one of six altitude/speed/throttle settings.

The sensors jump from one level to another with the regime, far more than they move with
wear. Scaling each sensor within its regime removes those jumps and leaves the wear visible.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# the three settings are noisy measurements of six set points: rounding finds the set points
ROUNDING = {"setting_1": 0, "setting_2": 2, "setting_3": 0}


def regime_of(df: pd.DataFrame) -> pd.Series:
    """The regime of each row, as text: "35.0_0.84_100.0" for settings 35, 0.84 and 100."""
    rounded = df[list(ROUNDING)].round(ROUNDING)
    # + 0.0 turns -0.0 (a small negative value rounded) into 0.0, so both give the same text
    return rounded.add(0.0).astype(str).agg("_".join, axis=1)


def fit_regime_scalers(train_df: pd.DataFrame, sensors: list[str]) -> dict[str, StandardScaler]:
    """One scaler per regime, learnt on the training rows of that regime only."""
    regimes = regime_of(train_df)
    return {
        regime: StandardScaler().fit(rows[sensors])
        for regime, rows in train_df.groupby(regimes, sort=True)
    }


def scale_by_regime(
    df: pd.DataFrame, scalers: dict[str, StandardScaler], sensors: list[str]
) -> pd.DataFrame:
    """Return a copy of df where each row's sensors are scaled with the scaler of its regime."""
    df = df.copy()
    df[sensors] = df[sensors].astype(float)  # integer sensors such as s_17 receive decimals
    regimes = regime_of(df)
    unknown = set(regimes) - set(scalers)
    if unknown:
        raise ValueError(f"regimes never seen in training: {sorted(unknown)}")
    for regime, scaler in scalers.items():
        rows = regimes == regime
        if rows.any():
            df.loc[rows, sensors] = scaler.transform(df.loc[rows, sensors])
    return df
