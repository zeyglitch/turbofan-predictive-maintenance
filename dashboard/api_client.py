"""The dashboard's only way to get a prediction: HTTP calls to the API of phase 10.

API_URL is the API's address (http://127.0.0.1:8000 by default; in Docker, http://api:8000).
"""

import os

import httpx
import pandas as pd
import streamlit as st


def api_url() -> str:
    return os.environ.get("API_URL", "http://127.0.0.1:8000")


def model_info(url: str) -> dict | None:
    """GET /model: the model the API serves, or None if the API does not answer."""
    try:
        response = httpx.get(f"{url}/model", timeout=2)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return response.json()


@st.cache_data(show_spinner="Asking the API for a prediction at every cycle...")
def prediction_curve(url: str, engine: pd.DataFrame, model_version: str) -> pd.DataFrame:
    """One POST /predict per cycle, each time with the history up to that cycle.

    It replays the engine's life as if its cycles arrived one by one. model_version is part
    of the cache key: when the API serves a new champion, the curve is asked again.
    """
    unit_id = int(engine["unit_id"].iloc[0])
    cycles = engine.sort_values("cycle").drop(columns="unit_id").to_dict(orient="records")
    answers = []
    with httpx.Client(base_url=url, timeout=10) as client:  # one connection for all the calls
        for n in range(1, len(cycles) + 1):
            response = client.post("/predict", json={"unit_id": unit_id, "cycles": cycles[:n]})
            response.raise_for_status()
            answers.append(response.json())
    return pd.DataFrame(answers)
