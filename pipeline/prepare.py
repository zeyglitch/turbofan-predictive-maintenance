"""Add the target, split engines, scale sensors and write the processed tables."""

import json

import joblib

from turbofan.config import INDEX_COLUMNS, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.data import load_raw, load_test_rul
from turbofan.labels import add_rul, add_test_rul
from turbofan.preprocessing import fit_scaler, scale_sensors, split_units


def main() -> None:
    params = load_params()
    subset = params["data"]["subset"]
    cap = params["prepare"]["rul_cap"]
    sensors = params["prepare"]["sensors"]
    split = params["split"]

    train_all = add_rul(load_raw(subset, "train"), cap)
    test = add_test_rul(load_raw(subset, "test"), load_test_rul(subset), cap)
    print(f"read {subset}: {len(train_all)} training rows, {len(test)} test rows")

    units = split_units(
        train_all["unit_id"].unique().tolist(), split["n_val"], split["n_calib"], split["seed"]
    )
    parts = {name: train_all[train_all["unit_id"].isin(ids)] for name, ids in units.items()}
    parts["test"] = test

    # the scaler only sees the 70 training engines
    scaler = fit_scaler(parts["train"], sensors)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = {"subset": subset, "rul_cap": cap, "sensors": sensors, "parts": {}}
    for name, df in parts.items():
        columns = INDEX_COLUMNS + sensors + ["rul", "rul_clipped"]
        if name == "test":
            columns = columns + ["final_rul"]
        scaled = scale_sensors(df, scaler, sensors)[columns]
        scaled.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        report["parts"][name] = {
            "n_engines": int(df["unit_id"].nunique()),
            "n_rows": len(df),
            "units": sorted(df["unit_id"].unique().tolist()),
        }
        print(f"wrote {name}.parquet: {report['parts'][name]['n_engines']} engines, {len(df)} rows")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "prepare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {MODELS_DIR / 'scaler.joblib'} and {REPORTS_DIR / 'prepare.json'}")


if __name__ == "__main__":
    main()
