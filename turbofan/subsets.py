"""Several C-MAPSS subsets as one fleet: FD001 + FD003 for the challenger of phase 14.

Each subset numbers its engines from 1, so engine 1 of FD003 would be mistaken for engine 1 of
FD001. The engines of the n-th subset of the list get 1000 * n added to their unit_id: with
[FD001, FD003], FD001 keeps 1 to 100 and FD003 becomes 1001 to 1100.
"""

import pandas as pd

from turbofan.config import RAW_DIR
from turbofan.data import load_raw, load_test_rul

UNIT_OFFSET = 1000  # more than the number of engines of any subset (at most 260)


def load_subsets(subsets: list[str], part: str, raw_dir=RAW_DIR) -> pd.DataFrame:
    """The train or test files of several subsets, one after the other, with unique unit_id."""
    tables = []
    for position, subset in enumerate(subsets):
        df = load_raw(subset, part, raw_dir)
        df["unit_id"] = df["unit_id"] + position * UNIT_OFFSET
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def load_subsets_test_rul(subsets: list[str], raw_dir=RAW_DIR) -> pd.Series:
    """The true remaining life of the test engines of several subsets, indexed by unique unit_id."""
    series = []
    for position, subset in enumerate(subsets):
        rul = load_test_rul(subset, raw_dir)
        rul.index = rul.index + position * UNIT_OFFSET
        series.append(rul)
    return pd.concat(series)
