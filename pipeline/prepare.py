"""Add the target, split engines, scale sensors and write the processed tables.

The DVC stage prepares params data.subset (FD001). The retraining flow (flows/) calls
prepare() with FD001 + FD003 and its own output folders. With regimes.normalize (FD002 and
FD004, six flight regimes), each sensor is scaled within its regime.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from turbofan.config import INDEX_COLUMNS, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.labels import add_rul, add_test_rul
from turbofan.preprocessing import fit_scaler, scale_sensors, split_units
from turbofan.regimes import fit_regime_scalers, scale_by_regime
from turbofan.subsets import UNIT_OFFSET, load_subsets, load_subsets_test_rul


def split_by_subset(train_all: pd.DataFrame, subsets: list[str], split: dict) -> dict:
    """Split the engines of each subset on its own (70/15/15 for 100 engines), then gather them."""
    units = {"train": [], "val": [], "calib": []}
    for position in range(len(subsets)):
        ids = train_all["unit_id"][train_all["unit_id"] // UNIT_OFFSET == position].unique()
        # a share, not a fixed number: a subset of 260 engines keeps 39 for validation, not 15
        n_val = round(len(ids) * split["val_share"])
        n_calib = round(len(ids) * split["calib_share"])
        own = split_units(ids.tolist(), n_val, n_calib, split["seed"])
        for name in units:
            units[name] += own[name]
    return {name: train_all[train_all["unit_id"].isin(ids)] for name, ids in units.items()}


def prepare(
    subsets: list[str], params: dict, processed_dir: Path, scaler_path: Path, report_path: Path
) -> dict:
    """Write train/val/calib/test parquet files, the scaler and the report; return the report."""
    cap = params["prepare"]["rul_cap"]
    sensors = params["prepare"]["sensors"]

    train_all = add_rul(load_subsets(subsets, "train"), cap)
    test = add_test_rul(load_subsets(subsets, "test"), load_subsets_test_rul(subsets), cap)
    name = "+".join(subsets)
    print(f"read {name}: {len(train_all)} training rows, {len(test)} test rows")

    parts = split_by_subset(train_all, subsets, params["split"])
    parts["test"] = test

    # the scaler only sees the training engines; with six flight regimes (FD002, FD004), one
    # scaler per regime, so that a change of regime is not mistaken for wear
    by_regime = params["regimes"]["normalize"]
    if by_regime:
        scaler = fit_regime_scalers(parts["train"], sensors)
    else:
        scaler = fit_scaler(parts["train"], sensors)

    processed_dir.mkdir(parents=True, exist_ok=True)
    report = {"subset": name, "rul_cap": cap, "sensors": sensors, "parts": {}}
    for part, df in parts.items():
        columns = INDEX_COLUMNS + sensors + ["rul", "rul_clipped"]
        if part == "test":
            columns = columns + ["final_rul"]
        scale = scale_by_regime if by_regime else scale_sensors
        scaled = scale(df, scaler, sensors)[columns]
        scaled.to_parquet(processed_dir / f"{part}.parquet", index=False)
        report["parts"][part] = {
            "n_engines": int(df["unit_id"].nunique()),
            "n_rows": len(df),
            "units": sorted(df["unit_id"].unique().tolist()),
        }
        print(f"wrote {part}.parquet: {report['parts'][part]['n_engines']} engines, {len(df)} rows")

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the same bytes on Windows and Linux, so the fingerprint in dvc.lock matches
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {scaler_path} and {report_path}")
    return report


def main() -> None:
    params = load_params()
    prepare(
        [params["data"]["subset"]],
        params,
        PROCESSED_DIR,
        MODELS_DIR / "scaler.joblib",
        REPORTS_DIR / "prepare.json",
    )


if __name__ == "__main__":
    main()
