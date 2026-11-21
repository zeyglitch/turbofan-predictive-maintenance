"""Engine view: one test engine, its remaining life predicted cycle after cycle by the API."""

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client, loaders

st.title("Engine view")
st.caption(
    "Pick a test engine: its life is replayed cycle by cycle, and at each cycle the API "
    "predicts how many cycles are left, with a 90 % interval. The dotted line is the truth, "
    "known here because NASA gives the remaining life of each test engine."
)

raw_file = loaders.path("data", "raw", "test_FD001.txt")
truth_file = loaders.path("data", "processed", "test.parquet")
ranking_file = loaders.path("reports", "explain", "sensor_ranking.csv")
loaders.require([raw_file, truth_file, ranking_file], "dvc repro")

url = api_client.api_url()
model = api_client.model_info(url)
if model is None:
    st.error(
        f"The dashboard asks the API for every prediction, and nothing answers at {url}. "
        "Start it in another terminal with `uvicorn api.main:app --port 8000`, "
        "then reload this page.",
        title="The prediction API does not answer",
        icon=":material/cloud_off:",
    )
    st.stop()

test = loaders.read_raw(loaders.params()["data"]["subset"], "test")
truth = loaders.read_parquet(truth_file)
threshold = model["alert_threshold"]
units = sorted(test["unit_id"].unique())
unit_id = st.selectbox("Test engine", units, index=units.index(34) if 34 in units else 0)

engine = test[test["unit_id"] == unit_id]
try:
    curve = api_client.prediction_curve(url, engine, model["model_version"])
except httpx.HTTPError as error:
    st.error(f"The API stopped answering during the replay: {error}", title="API error")
    st.stop()
# the truth at each cycle, capped like the training target
curve = curve.merge(truth[truth["unit_id"] == unit_id][["cycle", "rul_clipped"]], on="cycle")
last = curve.iloc[-1]

with st.container(horizontal=True):
    st.metric("Cycles seen", int(last["cycle"]), border=True)
    st.metric(
        "Predicted remaining life",
        f"{last['rul']:.0f} cycles",
        help=f"{model['model_type'].upper()} v{model['model_version']}, served by the API.",
        border=True,
    )
    st.metric("90 % interval", f"{last['rul_low']:.0f} – {last['rul_high']:.0f}", border=True)
    st.metric("True remaining life", f"{last['rul_clipped']:.0f} cycles", border=True)
    st.metric("Alert", "yes" if last["alert"] else "no", border=True)

figure = go.Figure(
    [
        go.Scatter(x=curve["cycle"], y=curve["rul_high"], line={"width": 0}, showlegend=False),
        go.Scatter(
            x=curve["cycle"],
            y=curve["rul_low"],
            fill="tonexty",  # fills down from the line above: the interval band
            fillcolor="rgba(31, 119, 180, 0.2)",
            line={"width": 0},
            name="90 % interval",
        ),
        go.Scatter(x=curve["cycle"], y=curve["rul"], line={"color": "#1f77b4"}, name="prediction"),
        go.Scatter(
            x=curve["cycle"],
            y=curve["rul_clipped"],
            line={"color": "#d62728", "dash": "dot"},
            name="truth (capped)",
        ),
    ]
)
figure.add_hline(y=threshold, line_dash="dash", annotation_text=f"alert threshold = {threshold}")
figure.update_layout(xaxis_title="cycle", yaxis_title="remaining life (cycles)")
st.plotly_chart(figure, alt=f"Predicted and true remaining life of test engine {unit_id}")
st.caption(
    "Early in life the history is short and the prediction stays near the cap: nothing in the "
    "sensors shows wear yet. Test engines stop before failure, at a cycle chosen by NASA, so the "
    "truth does not always reach 0. Each point is one API call; if the API was started with "
    "PREDICTION_LOG, they are all written to its log."
)

st.subheader("The sensors the model relies on most, on this engine")
method = {"lstm": "lstm_permutation", "xgboost": "xgboost_shap"}.get(model["model_type"])
if method is None:
    st.caption(
        f"No importance was computed for a {model['model_type']} model (see Explainability)."
    )
else:
    top = loaders.read_csv(ranking_file).nlargest(3, method)["sensor"].tolist()
    st.caption(
        f"The three most important sensors for the {model['model_type'].upper()} across the "
        f"validation engines ({', '.join(top)}; see Explainability), drawn on this engine. "
        "This is the model's general ranking, not an explanation of this engine alone."
    )
    long = engine.melt(id_vars="cycle", value_vars=top, var_name="sensor")
    sensors_figure = px.line(
        long, x="cycle", y="value", facet_col="sensor", facet_col_spacing=0.08, height=300
    )
    sensors_figure.update_yaxes(matches=None, showticklabels=True)
    sensors_figure.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(sensors_figure, alt=f"The three most important sensors of engine {unit_id}")
