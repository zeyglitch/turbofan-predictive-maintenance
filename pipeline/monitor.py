"""Monitoring report: the sensors logged by the API against a normal period of service, and the
real error of the predictions when the simulation has written the truth.

Run it after a simulation: python -m pipeline.monitor
It writes reports/monitoring/drift.json and drift.png.
"""

import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from turbofan.config import REPORTS_DIR, load_params
from turbofan.data import load_raw
from turbofan.metrics import rmse
from turbofan.monitoring import (
    MONITORING_DIR,
    build_drift_report,
    error_over_time,
    join_ground_truth,
    load_prediction_log,
    reference_period,
)

OUTPUT_DIR = REPORTS_DIR / "monitoring"


def load_reference(settings: dict) -> pd.DataFrame:
    """The reference period: every cycle of test FD001 engines 1 to reference_engines."""
    return reference_period(load_raw("FD001", "test"), settings["reference_engines"])


def plot_sensor_distributions(reference: pd.DataFrame, current: pd.DataFrame, report: dict):
    """One histogram per sensor, reference against current; the title says if it drifts."""
    rows = report["sensors"]
    n_lines = (len(rows) + 3) // 4  # four histograms per line
    fig, axes = plt.subplots(n_lines, 4, figsize=(14, 2.6 * n_lines))
    for ax, row in zip(axes.flat, rows, strict=False):
        sensor = row["sensor"]
        ax.hist(reference[sensor], bins=40, density=True, alpha=0.5, label="reference")
        ax.hist(current[sensor], bins=40, density=True, alpha=0.5, label="current")
        verdict = "DRIFT" if row["drifted"] else "ok"
        ax.set_title(
            f"{sensor}: {row['distance']:.2f} std, {verdict}",
            color="red" if row["drifted"] else "black",
        )
    for ax in axes.flat[len(rows) :]:
        ax.axis("off")  # empty cells of the last row
    axes.flat[0].legend()
    fig.tight_layout()
    return fig


def error_summary(log: pd.DataFrame, truth: pd.DataFrame, cap: int, settings: dict) -> dict:
    """RMSE of the predictions whose truth is known: overall, per half, and on the last window."""
    errors = error_over_time(join_ground_truth(log, truth), cap, settings["error_window"])
    first, second = errors.iloc[: len(errors) // 2], errors.iloc[len(errors) // 2 :]
    last_window = errors["rolling_rmse"].iloc[-1]
    return {
        "n_with_truth": len(errors),
        "rmse": rmse(errors["rul_true_clipped"], errors["rul"]),
        "rmse_first_half": rmse(first["rul_true_clipped"], first["rul"]),
        "rmse_second_half": rmse(second["rul_true_clipped"], second["rul"]),
        # empty while fewer than error_window predictions have a truth
        "last_window_rmse": None if pd.isna(last_window) else round(float(last_window), 2),
    }


def main() -> None:
    params = load_params()
    settings = params["monitoring"]
    sensors = params["prepare"]["sensors"]
    log_path = MONITORING_DIR / "predictions.jsonl"
    if not log_path.exists():
        raise SystemExit(f"no prediction log at {log_path}: start the API, then run a simulation")

    reference = load_reference(settings)
    current = load_prediction_log(log_path)
    report = build_drift_report(reference, current, sensors, settings["drift_threshold"])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reference": len(reference),
        "n_current": len(current),
        "drift_threshold": settings["drift_threshold"],
        "drift_share_threshold": settings["drift_share_threshold"],
        "retraining_suggested": report["drift_share"] >= settings["drift_share_threshold"],
        **report,
    }
    truth_path = MONITORING_DIR / "ground_truth.jsonl"
    if truth_path.exists() and truth_path.stat().st_size > 0:
        report["error"] = error_summary(
            current, load_prediction_log(truth_path), params["prepare"]["rul_cap"], settings
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure = plot_sensor_distributions(reference, current, report)
    figure.savefig(OUTPUT_DIR / "drift.png", dpi=100, bbox_inches="tight")
    plt.close(figure)

    print(f"{len(current)} logged predictions against {len(reference)} reference cycles")
    print(pd.DataFrame(report["sensors"]).set_index("sensor").round(3).to_string())
    print(
        f"drift share {report['drift_share']:.2f} ({len(report['drifted_sensors'])} of "
        f"{len(sensors)} sensors: {report['drifted_sensors']}); retraining suggested: "
        f"{report['retraining_suggested']}"
    )
    if "error" in report:
        error = report["error"]
        print(
            f"real error on {error['n_with_truth']} predictions: RMSE {error['rmse']:.2f} "
            f"(first half {error['rmse_first_half']:.2f}, second half "
            f"{error['rmse_second_half']:.2f}), last {settings['error_window']}: "
            f"{error['last_window_rmse']}"
        )
    print(f"wrote {OUTPUT_DIR / 'drift.json'} and drift.png")


if __name__ == "__main__":
    main()
