"""Prediction API: the champion model behind three routes.

Run it with `uvicorn api.main:app --port 8000`, then open http://127.0.0.1:8000/docs.
MODEL_DIR is the bundle folder (default models/champion). PREDICTION_LOG is the JSON Lines file
where every prediction is appended, for example monitoring/predictions.jsonl; without it, no log.
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from api.schemas import EngineHistory, Health, ModelInfo, Prediction
from turbofan.bundle import load_bundle
from turbofan.config import MODELS_DIR, SENSOR_COLUMNS
from turbofan.predict import predict_engine

state = {}  # filled once at start-up, read by every request


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the bundle once, when the server starts, instead of at every request."""
    state["bundle"] = load_bundle(Path(os.environ.get("MODEL_DIR", MODELS_DIR / "champion")))
    # no default: in PowerShell, setting a variable to "" deletes it, so "empty = off" can't work
    state["log_path"] = os.environ.get("PREDICTION_LOG", "")
    yield
    state.clear()


app = FastAPI(title="Turbofan RUL API", lifespan=lifespan)


@app.get("/health")
def health() -> Health:
    return Health(status="ok", model_loaded="bundle" in state)


@app.get("/model")
def model() -> ModelInfo:
    metadata = state["bundle"]["metadata"]
    return ModelInfo(model_version=metadata["registry_version"], **metadata)


@app.post("/predict")
def predict(history: EngineHistory) -> Prediction:
    metadata = state["bundle"]["metadata"]
    # the request becomes the same table as a raw file: one row per cycle, the raw columns
    engine_df = pd.DataFrame([cycle.model_dump() for cycle in history.cycles])
    engine_df.insert(0, "unit_id", history.unit_id)
    result = predict_engine(state["bundle"], engine_df)
    last = engine_df.loc[engine_df["cycle"].idxmax()]
    prediction = Prediction(
        unit_id=history.unit_id,
        cycle=int(last["cycle"]),
        model_version=metadata["registry_version"],
        **result,
    )
    if state["log_path"]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unit_id": prediction.unit_id,
            "cycle": prediction.cycle,
            **{sensor: float(last[sensor]) for sensor in SENSOR_COLUMNS},
            "rul": prediction.rul,
            "rul_low": prediction.rul_low,
            "rul_high": prediction.rul_high,
            "alert": prediction.alert,
            "model_version": prediction.model_version,
        }
        append_to_log(Path(state["log_path"]), record)
    return prediction


def append_to_log(path: Path, record: dict) -> None:
    """Add one line of JSON at the end of the log, creating the file and its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record) + "\n")
