"""Remaining useful life (RUL): the target the models learn to predict."""

import pandas as pd


def clip_rul(rul: pd.Series, cap: int) -> pd.Series:
    """Return the RUL with every value above cap replaced by cap."""
    return rul.clip(upper=cap)


def add_rul(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add rul and rul_clipped to training engines, which are followed until failure."""
    # the last cycle of each engine is its last cycle before failure, so its rul is 0 there
    last_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = last_cycle - df["cycle"]
    df["rul_clipped"] = clip_rul(df["rul"], cap)
    return df


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series, cap: int) -> pd.DataFrame:
    """Add final_rul, rul and rul_clipped to test engines, whose history stops before failure.

    final_rul is indexed by unit_id and gives the RUL at the last recorded cycle of each engine.
    """
    last_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["final_rul"] = df["unit_id"].map(final_rul)
    # an engine with final_rul 112 at its last row had 113 cycles left one cycle earlier, and so on
    df["rul"] = df["final_rul"] + last_cycle - df["cycle"]
    df["rul_clipped"] = clip_rul(df["rul"], cap)
    return df
