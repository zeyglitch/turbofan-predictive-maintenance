"""Monitoring of the model in service: data drift on its inputs, and its real error over time.

Data drift: each sensor is compared on its own, reference (a normal period of service) against
current (the last cycle of every request logged by the API). Real error: each logged prediction
against the true remaining life, when the simulation has written it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from turbofan.config import ROOT

MONITORING_DIR = ROOT / "monitoring"  # what the simulated production writes (not in git)


def load_prediction_log(path: Path) -> pd.DataFrame:
    """The API's log (JSON Lines) as a table: one row per prediction, in order of arrival."""
    return pd.read_json(path, lines=True)


def reference_period(test: pd.DataFrame, n_engines: int) -> pd.DataFrame:
    """The reference of the drift report: every cycle of test engines 1 to n_engines.

    Not the training engines: those are followed until failure, while engines in service are
    seen at every stage of their life, mostly before the end. Compared with full lives, a normal
    fleet already looks drifted (see the phase 13 notes).
    """
    return test[test["unit_id"] <= n_engines]


def build_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, sensors: list[str], threshold: float
) -> dict:
    """Compare each sensor's distribution; returns {"drift_share", "drifted_sensors", "sensors"}.

    distance: how far the values had to move to turn one distribution into the other
    (Wasserstein distance), in standard deviations of the reference, so that sensors measured
    in different units can share one threshold. A sensor drifts when distance >= threshold.
    The Kolmogorov-Smirnov test is kept for information: with thousands of rows, its p-value is
    tiny for differences too small to matter, so it does not decide.
    """
    rows = []
    for sensor in sensors:
        distance = (
            wasserstein_distance(reference[sensor], current[sensor]) / reference[sensor].std()
        )
        ks = ks_2samp(reference[sensor], current[sensor])
        rows.append(
            {
                "sensor": sensor,
                "reference_mean": float(reference[sensor].mean()),
                "current_mean": float(current[sensor].mean()),
                "distance": float(distance),
                "drifted": bool(distance >= threshold),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    drifted = [row["sensor"] for row in rows if row["drifted"]]
    return {
        "drift_share": len(drifted) / len(sensors),
        "drifted_sensors": drifted,
        "sensors": rows,
    }


def join_ground_truth(log: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """The logged predictions whose true remaining life is known, still in order of arrival.

    Predictions without a truth (an engine replayed by the dashboard, for example) are left out.
    """
    truth = truth.drop_duplicates(["unit_id", "cycle"], keep="last")
    # an inner merge keeps the order of the left table: the order of arrival
    return log.merge(truth[["unit_id", "cycle", "rul_true"]], on=["unit_id", "cycle"])


def error_over_time(joined: pd.DataFrame, cap: int, window: int) -> pd.DataFrame:
    """Add the error of each prediction and the RMSE of the last `window` predictions.

    The truth is capped like the training target: above `cap` cycles, the model is not asked
    to tell 150 from 200. rolling_rmse is empty until `window` predictions have arrived.
    """
    df = joined.reset_index(drop=True)
    df["rul_true_clipped"] = df["rul_true"].clip(upper=cap)
    df["error"] = df["rul"] - df["rul_true_clipped"]
    df["rolling_rmse"] = np.sqrt((df["error"] ** 2).rolling(window).mean())
    return df
