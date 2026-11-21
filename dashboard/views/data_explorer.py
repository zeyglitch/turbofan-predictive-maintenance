"""Data explorer: the raw NASA files, from the whole fleet down to one engine."""

import plotly.express as px
import streamlit as st

from dashboard import loaders
from turbofan.config import SENSOR_COLUMNS
from turbofan.describe import engine_lifetimes, sensor_summary

st.title("Data explorer")
st.caption(
    "The raw training files, as NASA distributes them: one row per engine and per flight cycle, "
    "3 operating settings and 21 sensors. Each engine is followed until it fails. "
    "Start with the fleet, then pick one engine."
)

subset = st.segmented_control(
    "Subset", ["FD001", "FD002", "FD003", "FD004"], default="FD001", required=True
)
if subset != "FD001":
    st.caption(
        f"The models use FD001 only. {subset} is shown for comparison: the kept sensors below "
        "are those chosen on FD001."
    )
loaders.require([loaders.path("data", "raw", f"train_{subset}.txt")], "python -m pipeline.download")
train = loaders.read_raw(subset, "train")
kept = loaders.params()["prepare"]["sensors"]

st.subheader("The fleet")
lifetimes = engine_lifetimes(train)
st.caption(
    f"{len(lifetimes)} engines, from {lifetimes.min()} to {lifetimes.max()} cycles "
    f"(median {lifetimes.median():.0f}). Every engine starts healthy, with its own wear, "
    "and runs until failure: its lifetime is its last cycle."
)
st.plotly_chart(
    px.histogram(lifetimes, x="lifetime", nbins=30, labels={"lifetime": "Lifetime (cycles)"}),
    alt="Histogram of the engines' lifetimes",
)

st.subheader("Which sensors carry information")
st.caption(
    "A sensor that takes at most two values over the whole file cannot follow the wear: "
    "it is dropped. The others are scaled and given to the models."
)
st.dataframe(
    sensor_summary(train, kept),
    column_config={
        "std": st.column_config.NumberColumn("Standard deviation", format="%.4f"),
        "distinct_values": "Distinct values",
        "kept": "Used",
        "reason": "Why",
    },
)

st.subheader("Correlations between the kept sensors")
st.caption(
    "Red: two sensors rise together; blue: one rises when the other falls. Sensors that follow "
    "the same wear move together, so they partly repeat the same information."
    + (" On FD001, s_9 and s_14 form a pair of their own." if subset == "FD001" else "")
)
st.plotly_chart(
    px.imshow(
        train[kept].corr(),
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    ),
    alt="Correlation matrix of the kept sensors",
)

st.subheader("One engine")
unit_id = st.selectbox("Engine", sorted(train["unit_id"].unique()))
engine = train[train["unit_id"] == unit_id]
st.caption(
    f"Engine {unit_id} runs {len(engine)} cycles. Grey curves: dropped sensors."
    + (
        " On FD001, the wear only rises above the noise in the last 100 to 150 cycles."
        if subset == "FD001"
        else ""
    )
)
long = engine.melt(id_vars="cycle", value_vars=SENSOR_COLUMNS, var_name="sensor")
long["used"] = long["sensor"].isin(kept).map({True: "used", False: "dropped"})
figure = px.line(
    long,
    x="cycle",
    y="value",
    facet_col="sensor",
    facet_col_wrap=3,
    facet_row_spacing=0.03,
    facet_col_spacing=0.07,  # room for each small chart's own y-axis labels
    color="used",
    color_discrete_map={"used": "#1f77b4", "dropped": "#aaaaaa"},
    height=1400,
)
figure.update_yaxes(matches=None, showticklabels=True, title=None)
figure.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(figure, alt=f"The 21 sensors of engine {unit_id} over its life")

with st.expander("Raw rows of this engine"):
    st.dataframe(engine, hide_index=True)
