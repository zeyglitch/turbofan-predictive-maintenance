"""Data drift: do the sensors received in production still look like the ones the model learnt on?

Each sensor is compared on its own, reference (a normal period of service) against current
(the last cycle of every request logged by the API).
"""

from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from turbofan.config import ROOT

MONITORING_DIR = ROOT / "monitoring"  # what the simulated production writes (not in git)


def load_prediction_log(path: Path) -> pd.DataFrame:
    """The API's log (JSON Lines) as a table: one row per prediction, in order of arrival."""
    return pd.read_json(path, lines=True)


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
