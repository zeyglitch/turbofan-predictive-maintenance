"""Data drift report: the sensors logged by the API against a normal period of service.

Run it after a simulation: python -m pipeline.monitor
It writes reports/monitoring/drift.json and drift.png.
"""

import json
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd

from turbofan.config import REPORTS_DIR, load_params
from turbofan.data import load_raw
from turbofan.monitoring import MONITORING_DIR, build_drift_report, load_prediction_log

OUTPUT_DIR = REPORTS_DIR / "monitoring"


def load_reference(settings: dict) -> pd.DataFrame:
    """The reference period: every cycle of test FD001 engines 1 to reference_engines.

    Not the training engines: those are followed until failure, while engines in service are
    seen at every stage of their life, mostly before the end. Compared with full lives, a normal
    fleet already looks drifted (see the phase 13 notes).
    """
    test = load_raw("FD001", "test")
    return test[test["unit_id"] <= settings["reference_engines"]]


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
    print(f"wrote {OUTPUT_DIR / 'drift.json'} and drift.png")


if __name__ == "__main__":
    main()
