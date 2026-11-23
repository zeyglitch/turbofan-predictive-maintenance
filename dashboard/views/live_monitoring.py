"""Live monitoring: what the API has received, compared with a normal period of service."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import loaders
from turbofan.monitoring import (
    build_drift_report,
    error_over_time,
    join_ground_truth,
    reference_period,
)

st.title("Live monitoring")
st.caption(
    "The model in service: every prediction the API logs, compared with a normal period of "
    "service (test engines 1 to 50 of FD001). A sensor drifts when its values have moved by "
    "0.1 standard deviation or more. The real error is known here only because the simulation "
    "writes the truth; in real life it arrives after the failure."
)
with st.expander("How to run a scenario"):
    st.markdown(
        "In a first terminal, start the API with its log (or use `docker compose up`):\n\n"
        "```powershell\n"
        '$env:PREDICTION_LOG = "monitoring/predictions.jsonl"\n'
        "uvicorn api.main:app --port 8000\n"
        "```\n\n"
        "In a second terminal, replay engines: `normal`, `sensor_offset` or `new_fault_mode`.\n\n"
        "```powershell\n"
        "python -m pipeline.simulate --scenario sensor_offset --step 5\n"
        "```\n\n"
        "Empty the `monitoring/` folder before switching scenarios. Engines replayed on the "
        "Engine view page are logged too, without a truth."
    )

log_file = loaders.path("monitoring", "predictions.jsonl")
truth_file = loaders.path("monitoring", "ground_truth.jsonl")
raw_file = loaders.path("data", "raw", "test_FD001.txt")
loaders.require([raw_file], "python -m pipeline.download")

params = loaders.params()
settings = params["monitoring"]
sensors = params["prepare"]["sensors"]
reference = reference_period(loaders.read_raw("FD001", "test"), settings["reference_engines"])
live = st.toggle("Refresh every 5 seconds", help="Turn on while a simulation is running.")


@st.fragment(run_every="5s" if live else None)
def live_view() -> None:
    """Everything that changes while predictions arrive; reruns alone when refreshing."""
    # checked here and not with loaders.require, so that the page waits for the first prediction
    if not log_file.exists() or log_file.stat().st_size == 0:
        st.info(
            "No prediction logged yet. Start the API with its log, then run "
            "`python -m pipeline.simulate --scenario normal` (see above). With the refresh "
            "turned on, this page shows the predictions as they arrive.",
            title="Waiting for the simulated production",
            icon=":material/hourglass:",
        )
        return
    log = loaders.read_jsonl(log_file)
    report = build_drift_report(reference, log, sensors, settings["drift_threshold"])
    scenario, errors = "unknown", pd.DataFrame()
    if truth_file.exists() and truth_file.stat().st_size > 0:
        truth = loaders.read_jsonl(truth_file)
        scenario = truth["scenario"].iloc[-1]
        errors = error_over_time(
            join_ground_truth(log, truth), params["prepare"]["rul_cap"], settings["error_window"]
        )

    with st.container(horizontal=True):
        st.metric("Predictions logged", f"{len(log):,}", border=True)
        st.metric("Engines", log["unit_id"].nunique(), border=True)
        st.metric(
            "Scenario", scenario, help="Written by the simulation next to the truth.", border=True
        )
        st.metric(
            "Drift share",
            f"{report['drift_share']:.2f}",
            help=(
                f"{len(report['drifted_sensors'])} of {len(sensors)} sensors drift "
                f"({', '.join(report['drifted_sensors']) or 'none'}). From "
                f"{settings['drift_share_threshold']}, retraining is suggested."
            ),
            border=True,
        )
        # empty without a truth, or while fewer than error_window predictions have one
        last_rmse = errors["rolling_rmse"].iloc[-1] if len(errors) else None
        st.metric(
            f"RMSE, last {settings['error_window']} predictions",
            "n/a" if pd.isna(last_rmse) else f"{last_rmse:.1f} cycles",
            help="Against the true remaining life, capped at the training cap.",
            border=True,
        )
    if report["drift_share"] >= settings["drift_share_threshold"]:
        st.warning(
            f"{report['drift_share']:.0%} of the sensors drift: the model receives data unlike "
            "the normal period. Retraining is suggested (python -m flows.retrain_flow).",
            title="Retraining suggested",
            icon=":material/warning:",
        )
    st.subheader("Real error over time")
    if errors.empty:
        st.caption("No truth for these predictions: run a scenario with `pipeline.simulate`.")
    else:
        rolling = errors.assign(prediction=range(1, len(errors) + 1))
        error_figure = px.line(
            rolling, x="prediction", y="rolling_rmse", labels={"rolling_rmse": "RMSE (cycles)"}
        )
        st.plotly_chart(error_figure, alt="RMSE of the last predictions, in order of arrival")
        st.caption(
            f"Each point is the RMSE of the {settings['error_window']} predictions before it, in "
            "order of arrival; engines are replayed one after the other, so the second half of the "
            "curve is the second half of the fleet. It covers every cycle, not only the last one "
            "of each engine, so it is not the test RMSE of the Models page."
        )

    st.subheader("Drift, sensor by sensor")
    table = pd.DataFrame(report["sensors"])
    drift_figure = px.bar(
        table,
        x="sensor",
        y="distance",
        color="drifted",
        color_discrete_map={False: "#1f77b4", True: "#d62728"},
        # without it, plotly draws the drifted sensors first and the others after them
        category_orders={"sensor": sensors},
        labels={"distance": "distance (reference standard deviations)"},
    )
    drift_figure.add_hline(
        y=settings["drift_threshold"],
        line_dash="dash",
        annotation_text=f"threshold = {settings['drift_threshold']}",
    )
    st.plotly_chart(drift_figure, alt="Drift distance of each sensor against the threshold")
    st.caption(
        "Wasserstein distance between the reference and the logged values, divided by the "
        "reference standard deviation. The Kolmogorov-Smirnov p-value in the table is for "
        "comparison only: with thousands of rows, it is tiny even for differences that do not "
        "matter."
    )

    by_distance = table.sort_values("distance", ascending=False)["sensor"].tolist()
    sensor = st.selectbox("Compare one sensor", by_distance, key="sensor")
    both = pd.concat(
        [
            reference[[sensor]].assign(source="reference"),
            log[[sensor]].assign(source="logged by the API"),
        ]
    )
    histogram = px.histogram(
        both,
        x=sensor,
        color="source",
        barmode="overlay",  # drawn on top of each other, not side by side
        histnorm="probability density",  # same scale, whatever the number of rows
        color_discrete_sequence=["#9ecae1", "#d62728"],  # light blue, red: as on Preprocessing
    )
    st.plotly_chart(histogram, alt=f"Distribution of {sensor}, reference against logged values")
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "reference_mean": st.column_config.NumberColumn("Reference mean", format="%.3f"),
            "current_mean": st.column_config.NumberColumn("Logged mean", format="%.3f"),
            "distance": st.column_config.NumberColumn("Distance", format="%.3f"),
            "ks_statistic": st.column_config.NumberColumn("KS statistic", format="%.3f"),
            "ks_pvalue": st.column_config.NumberColumn("KS p-value", format="%.4f"),
        },
    )


live_view()
