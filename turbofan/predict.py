"""From the raw history of one engine to its remaining life: the function shared by the API,
the simulation and the dashboard, so that they all prepare the data exactly like the training."""

import numpy as np
import pandas as pd

from turbofan.classic_models import predict
from turbofan.conformal import prediction_interval
from turbofan.features import add_rolling_features, feature_columns
from turbofan.preprocessing import scale_sensors
from turbofan.training import predict_network
from turbofan.windows import last_window_per_unit

NETWORKS = ["lstm", "cnn"]


def predict_engine(bundle: dict, engine_df: pd.DataFrame) -> dict:
    """Raw history of one engine (raw columns, any length) -> prediction at its last cycle.

    Returns {"rul", "rul_low", "rul_high", "alert"}.
    """
    metadata = bundle["metadata"]
    sensors = metadata["sensors"]
    cap = metadata["rul_cap"]
    # the same scaling as in prepare, with the scaler learnt on the training engines
    df = scale_sensors(engine_df.sort_values("cycle"), bundle["scaler"], sensors)
    if metadata["model_type"] in NETWORKS:
        # one window ending on the last cycle, padded at the start if the history is short
        X, _ = last_window_per_unit(df, sensors, metadata["window_length"])
        rul = predict_network(bundle["model"], X, cap)[0]
    else:
        df = add_rolling_features(df, sensors, metadata["rolling_window"])
        last_row = df.iloc[[-1]]  # double brackets: a one-row table, not a Series
        rul = np.clip(predict(bundle["model"], last_row[feature_columns(sensors)])[0], 0, cap)
    low, high = prediction_interval([rul], metadata["conformal_q"], cap)
    return {
        "rul": float(rul),
        "rul_low": float(low[0]),
        "rul_high": float(high[0]),
        "alert": bool(rul <= metadata["alert_threshold"]),
    }
