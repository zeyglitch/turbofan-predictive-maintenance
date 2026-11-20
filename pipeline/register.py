"""Pick the champion on validation RMSE, register it in MLflow and write the delivered bundle."""

import json
import shutil
from datetime import datetime

import joblib
import pandas as pd

from pipeline.evaluate import load_trained
from turbofan.bundle import save_bundle
from turbofan.config import FEATURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.features import feature_columns
from turbofan.tracking import git_state, log_metrics, log_model, register_champion, start_run
from turbofan.windows import make_windows

# the baseline is a reference to beat, not a model to deliver
CANDIDATES = ["ridge", "random_forest", "xgboost", "lstm", "cnn"]
FLAVOURS = {
    "ridge": "sklearn",
    "random_forest": "sklearn",
    "xgboost": "xgboost",
    "lstm": "pytorch",
    "cnn": "pytorch",
}
MODEL_NAME = "turbofan-rul"


def read_metrics(file_name: str) -> dict:
    return json.loads((REPORTS_DIR / "metrics" / file_name).read_text(encoding="utf-8"))


def main() -> None:
    params = load_params()
    sensors = params["prepare"]["sensors"]
    val = read_metrics("val_classic.json") | read_metrics("val_deep.json")
    test = read_metrics("test.json")

    # chosen on validation only: the test scores are reported, they never decide
    ranking = sorted(CANDIDATES, key=lambda name: val[name]["rmse"])
    champion = ranking[0]
    print("validation rmse: " + ", ".join(f"{name} {val[name]['rmse']:.2f}" for name in ranking))
    print(f"champion: {champion} (test rmse {test[champion]['rmse']:.2f})")

    model = load_trained(champion, params)
    if champion in ("lstm", "cnn"):
        val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
        input_example = make_windows(val_df, sensors, params["windows"]["length"])[0][:5]
    else:
        input_example = pd.read_parquet(FEATURES_DIR / "val.parquet")[feature_columns(sensors)]
        input_example = input_example.head(5)

    with start_run(f"register_{champion}", params[champion], tags={"alias": "champion"}) as run:
        log_metrics(
            {
                "val_rmse": val[champion]["rmse"],
                "test_rmse": test[champion]["rmse"],
                "test_coverage": test[champion]["coverage"],
            }
        )
        log_model(model, FLAVOURS[champion], input_example)
    version = register_champion(run.info.run_id, MODEL_NAME, "champion")
    print(f"registered {MODEL_NAME} version {version} with alias champion")

    metadata = {
        "model_params": params[champion],
        "subsets": [params["data"]["subset"]],
        "sensors": sensors,
        "window_length": params["windows"]["length"],
        "rolling_window": params["features"]["rolling_window"],
        "rul_cap": params["prepare"]["rul_cap"],
        "conformal_q": test[champion]["conformal_q"],
        "alert_threshold": params["alert"]["threshold"],
        "val_rmse": val[champion]["rmse"],
        "test_rmse": test[champion]["rmse"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_state()["git_commit"],
        "mlflow_run_id": run.info.run_id,
        "registry_version": version,
    }
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    # start from an empty folder: DVC keeps it between runs (persist), and a previous champion
    # of another type would leave its model file behind
    shutil.rmtree(MODELS_DIR / "champion", ignore_errors=True)
    save_bundle(MODELS_DIR / "champion", champion, model, scaler, metadata)

    report = {
        "champion": champion,
        "val_rmse": {name: val[name]["rmse"] for name in ranking},
        "test_rmse": test[champion]["rmse"],
        "conformal_q": test[champion]["conformal_q"],
        "test_coverage": test[champion]["coverage"],
        "registry_version": version,
    }
    (REPORTS_DIR / "champion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"wrote {MODELS_DIR / 'champion'} and {REPORTS_DIR / 'champion.json'}")


if __name__ == "__main__":
    main()
