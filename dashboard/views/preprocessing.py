"""Preprocessing: four before/after views of what `prepare` and `featurize` do to the data."""

import plotly.express as px
import streamlit as st

from dashboard import loaders
from turbofan.describe import engine_lifetimes

TWO_COLOURS = ["#9ecae1", "#d62728"]  # light blue for "before", red for "after"

st.title("Preprocessing")
st.caption(
    "What happens to the raw files before a model sees them. Each section shows the data "
    "before and after one step, on one engine of the training set."
)

prepare_file = loaders.path("reports", "prepare.json")
processed_file = loaders.path("data", "processed", "train.parquet")
features_file = loaders.path("data", "features", "train.parquet")
raw_file = loaders.path("data", "raw", "train_FD001.txt")
loaders.require([prepare_file, processed_file, features_file, raw_file], "dvc repro")

params = loaders.params()
prepare = loaders.read_json(prepare_file)
raw = loaders.read_raw(params["data"]["subset"], "train")
processed = loaders.read_parquet(processed_file)
features = loaders.read_parquet(features_file)
cap = params["prepare"]["rul_cap"]
window = params["features"]["rolling_window"]

with st.container(horizontal=True, vertical_alignment="bottom"):
    unit_id = st.selectbox("Training engine", prepare["parts"]["train"]["units"])
    sensors = params["prepare"]["sensors"]
    sensor = st.selectbox("Sensor", sensors, index=sensors.index("s_11"))

st.subheader("1. Scaling")
st.caption(
    "Each sensor has its own unit and range (s_11 around 47, s_9 around 9,000). Scaling "
    "subtracts the sensor's mean and divides by its standard deviation, so that every sensor "
    "is centred on 0 with a spread of 1. The mean and spread are learnt on the 70 training "
    "engines only: validation, calibration and test engines are scaled with them, never the "
    "other way round."
)
before, after = st.columns(2)
raw_engine = raw[raw["unit_id"] == unit_id]
scaled_engine = processed[processed["unit_id"] == unit_id]
before.plotly_chart(
    px.line(raw_engine, x="cycle", y=sensor, title="Before: raw value"),
    alt=f"Raw {sensor} of engine {unit_id}",
)
after.plotly_chart(
    px.line(scaled_engine, x="cycle", y=sensor, title="After: scaled value"),
    alt=f"Scaled {sensor} of engine {unit_id}",
)

st.subheader("2. The target, capped")
st.caption(
    f"The target is the number of cycles left: it falls by 1 at each cycle and reaches 0 at "
    f"failure. Early in life, nothing in the sensors tells 300 cycles left from 200, so the "
    f"target is capped at {cap}: the model only has to say 'healthy' until the wear shows."
)
target = scaled_engine.melt(
    id_vars="cycle",
    value_vars=["rul", "rul_clipped"],
    var_name="target",
    value_name="cycles left",
)
target["target"] = target["target"].map({"rul": "before: linear", "rul_clipped": "after: capped"})
figure = px.line(
    target, x="cycle", y="cycles left", color="target", color_discrete_sequence=TWO_COLOURS
)
figure.add_hline(y=cap, line_dash="dot", annotation_text=f"cap = {cap}")
st.plotly_chart(figure, alt=f"Linear and capped remaining life of engine {unit_id}")

st.subheader("3. The split, engine by engine")
st.caption(
    "The 100 training engines are shared out whole: an engine is either in train, validation "
    "or calibration, never cut in two. If the rows of one engine were in train and validation, "
    "the model would be tested on an engine it has already seen. One point per engine."
)
part_of = {
    unit: part for part in ("train", "val", "calib") for unit in prepare["parts"][part]["units"]
}
split = engine_lifetimes(raw).reset_index()
split["part"] = split["unit_id"].map(part_of)
st.plotly_chart(
    px.scatter(
        split,
        x="unit_id",
        y="lifetime",
        color="part",
        labels={"unit_id": "Engine", "lifetime": "Lifetime (cycles)"},
    ),
    alt="The 100 training engines coloured by the part they were assigned to",
)

st.subheader("4. A rolling window")
st.caption(
    f"The classic models (Ridge, random forest, XGBoost) see one row at a time, with no memory. "
    f"To give them the recent past, each row receives the mean and spread of the sensor over the "
    f"last {window} cycles of the same engine. Move the slider: the mean of the shaded cycles is "
    f"the value of {sensor}_mean at the right edge of the band."
)
engine_features = features[features["unit_id"] == unit_id]
last_cycle = int(engine_features["cycle"].max())
end = st.slider("Last cycle of the window", 1, last_cycle, min(100, last_cycle))
start = max(1, end - window + 1)  # at the start of life, the window holds fewer cycles
curves = engine_features.melt(
    id_vars="cycle", value_vars=[sensor, f"{sensor}_mean"], var_name="curve", value_name="value"
)
figure = px.line(curves, x="cycle", y="value", color="curve", color_discrete_sequence=TWO_COLOURS)
figure.add_vrect(x0=start, x1=end, opacity=0.15, line_width=0, fillcolor="orange")
point = engine_features[engine_features["cycle"] == end]
figure.add_scatter(
    x=point["cycle"],
    y=point[f"{sensor}_mean"],
    mode="markers",
    marker={"size": 12, "color": "orange"},
    name=f"{sensor}_mean at cycle {end}",
)
st.plotly_chart(figure, alt=f"{sensor} of engine {unit_id} and its rolling mean")
window_values = engine_features[engine_features["cycle"].between(start, end)][sensor]
st.caption(
    f"Cycles {start} to {end}: mean of the {len(window_values)} values = "
    f"{window_values.mean():.3f}, and {sensor}_mean at cycle {end} = "
    f"{point[f'{sensor}_mean'].iloc[0]:.3f}."
)
