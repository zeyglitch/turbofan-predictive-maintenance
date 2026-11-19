"""Add rolling means and standard deviations of the sensors to the processed tables."""

import pandas as pd

from turbofan.config import FEATURES_DIR, PROCESSED_DIR, load_params
from turbofan.features import add_rolling_features, feature_columns

PARTS = ["train", "val", "calib", "test"]


def main() -> None:
    params = load_params()
    sensors = params["prepare"]["sensors"]
    window = params["features"]["rolling_window"]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
        df = add_rolling_features(df, sensors, window)
        df.to_parquet(FEATURES_DIR / f"{name}.parquet", index=False)
        print(f"wrote {name}.parquet: {len(df)} rows, {len(feature_columns(sensors))} features")


if __name__ == "__main__":
    main()
