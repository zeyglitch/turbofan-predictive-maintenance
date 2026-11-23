"""Retraining: each run of the retraining flow, its trigger, the scores and the decision."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import loaders

st.title("Retraining")
st.caption(
    "Each run of `python -m flows.retrain_flow`: the drift that triggered it, the champion and "
    "the challenger (same model, trained on FD001 + FD003) scored on the validation engines of "
    "both subsets, which decide, and on their test engines, which are only reported; then the "
    "decision of the written promotion rule."
)

report_dir = loaders.path("reports", "retraining")
reports = sorted(report_dir.glob("*.json")) if report_dir.exists() else []
if not reports:
    st.info(
        "No retraining run yet. Run `python -m flows.retrain_flow` in the project folder "
        "after a simulation (Live monitoring explains how), then reload this page.",
        title="Nothing to show yet",
        icon=":material/info:",
    )
    st.stop()

settings = loaders.params()["retraining"] | loaders.params()["monitoring"]
runs = [loaders.read_json(file) for file in reports]
latest = runs[-1]
decision = latest["decision"]
if not decision["retrain"]:
    verdict = "no retraining"
elif decision["promote"]:
    verdict = "challenger promoted"
else:
    verdict = "champion kept"

st.subheader(f"Latest run, {latest['started_at']}")
with st.container(horizontal=True):
    st.metric(
        "Drift share",
        f"{latest['drift_share']:.2f}",
        help=f"On {latest['n_logged']:,} logged predictions; retraining from "
        f"{settings['drift_share_threshold']}.",
        border=True,
    )
    st.metric("Champion before the run", f"v{latest['champion_version']}", border=True)
    st.metric(
        "Challenger",
        f"v{latest['challenger_version']}" if decision["retrain"] else "not trained",
        border=True,
    )
    st.metric("Decision", verdict, border=True)
if decision["promote"]:
    st.success(decision["reason"], title="Promoted", icon=":material/check_circle:")
else:
    st.info(decision["reason"], title="Not promoted", icon=":material/info:")

if decision["retrain"]:
    st.subheader("Champion against challenger")
    if "validation" in latest["scores"]:
        parts = [
            ("validation", "validation engines (they decide)"),
            ("test", "test engines (reported only)"),
        ]
        tables = {part: latest["scores"][part] for part, _ in parts}
    else:
        # runs written before the change: the test engines decided
        parts = [("test", "test engines (they decided, before the change)")]
        tables = {"test": latest["scores"]}
    for part, title in parts:
        scores = pd.DataFrame(tables[part]).rename_axis("engines").reset_index()
        scores["change"] = scores["challenger"] / scores["champion"] - 1
        st.markdown(f"**RMSE on the {title}**")
        long = scores.melt(
            id_vars="engines", value_vars=["champion", "challenger"], var_name="model"
        )
        figure = px.bar(
            long,
            x="engines",
            y="value",
            color="model",
            barmode="group",
            color_discrete_sequence=["#9ecae1", "#d62728"],  # light blue, red: before and after
            labels={"value": f"{part} RMSE (cycles)"},
        )
        st.plotly_chart(figure, alt=f"RMSE of the champion and the challenger, {title}")
        st.dataframe(
            scores,
            hide_index=True,
            column_config={
                "champion": st.column_config.NumberColumn("Champion RMSE", format="%.2f"),
                "challenger": st.column_config.NumberColumn("Challenger RMSE", format="%.2f"),
                "change": st.column_config.NumberColumn("Change", format="percent"),
            },
        )
    st.caption(
        "Validation: every cycle of the 15 validation engines of each subset, engines neither "
        "model was trained on. Test: the last cycle of each test engine, as in the pipeline. "
        '"all": both subsets together. The rule, on validation: promote only if "all" improves '
        f"by at least {settings['min_improvement']:.0%} and FD001 gets worse by less than "
        f"{settings['max_degradation']:.0%}. The test engines stay out of the decision, so their "
        "scores remain an honest measure after a promotion."
    )

st.subheader("Every run")
history = pd.DataFrame(
    [
        {
            "started_at": run["started_at"],
            "drift_share": run["drift_share"],
            "champion": run["champion_version"],
            "challenger": run.get("challenger_version", ""),
            "promoted": run["decision"]["promote"],
            "reason": run["decision"]["reason"],
        }
        for run in runs
    ]
)
st.dataframe(
    history.iloc[::-1],  # the latest first
    hide_index=True,
    column_config={"drift_share": st.column_config.NumberColumn("Drift share", format="%.2f")},
)
