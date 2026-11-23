"""Add rolling means and standard deviations of the sensors to the processed tables.

The DVC stage reads data/processed. featurize() takes its folders as arguments, so that another
preparation (FD001 + FD003, retraining) can get the same features; the retraining flow does not
need it yet, since its champion is a network.
"""

from pathlib import Path

import pandas as pd

from turbofan.config import FEATURES_DIR, PROCESSED_DIR, load_params
from turbofan.features import add_rolling_features, feature_columns

PARTS = ["train", "val", "calib", "test"]


def featurize(params: dict, processed_dir: Path, features_dir: Path) -> None:
    sensors = params["prepare"]["sensors"]
    window = params["features"]["rolling_window"]

    features_dir.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        df = pd.read_parquet(processed_dir / f"{name}.parquet")
        df = add_rolling_features(df, sensors, window)
        df.to_parquet(features_dir / f"{name}.parquet", index=False)
        print(f"wrote {name}.parquet: {len(df)} rows, {len(feature_columns(sensors))} features")


def main() -> None:
    featurize(load_params(), PROCESSED_DIR, FEATURES_DIR)


if __name__ == "__main__":
    main()
