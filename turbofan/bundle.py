"""The delivered model: one folder with the model, the scaler and what is needed to use them.

models/champion/
    model.pt | model.json | model.joblib
    scaler.joblib
    metadata.json
"""

import json
from pathlib import Path

import joblib
import torch
from xgboost import XGBRegressor

from turbofan.networks import build_network

METADATA_KEYS = [
    "model_type",
    "model_params",
    "subsets",
    "sensors",
    "window_length",
    "rolling_window",
    "rul_cap",
    "conformal_q",
    "alert_threshold",
    "val_rmse",
    "test_rmse",
    "created_at",
    "git_commit",
    "mlflow_run_id",
    "registry_version",
]
NETWORKS = ["lstm", "cnn"]


def model_file(model_type: str) -> str:
    """Name of the model file in the bundle, which depends on how its library saves it."""
    if model_type in NETWORKS:
        return "model.pt"
    if model_type in ("xgboost", "baseline"):
        return "model.json"
    return "model.joblib"


def save_bundle(directory: Path, model_type: str, model, scaler, metadata: dict) -> None:
    """Write the model, the scaler and metadata.json, which must hold every key of METADATA_KEYS."""
    metadata = metadata | {"model_type": model_type}
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata is missing {missing}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / model_file(model_type)
    if model_type in NETWORKS:
        torch.save(model.cpu().state_dict(), path)
    elif model_type == "xgboost":
        model.save_model(path)
    elif model_type == "baseline":
        path.write_text(json.dumps({"value": model}), encoding="utf-8")
    else:
        joblib.dump(model, path)
    joblib.dump(scaler, directory / "scaler.joblib")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8", newline="\n"
    )


def load_bundle(directory: Path) -> dict:
    """Read a bundle back: {"model", "scaler", "metadata"}. No MLflow needed."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_type = metadata["model_type"]
    path = directory / model_file(model_type)
    if model_type in NETWORKS:
        model = build_network(model_type, len(metadata["sensors"]), metadata["model_params"])
        model.load_state_dict(torch.load(path))
    elif model_type == "xgboost":
        model = XGBRegressor()
        model.load_model(path)
    elif model_type == "baseline":
        model = json.loads(path.read_text(encoding="utf-8"))["value"]
    else:
        model = joblib.load(path)
    return {
        "model": model,
        "scaler": joblib.load(directory / "scaler.joblib"),
        "metadata": metadata,
    }
