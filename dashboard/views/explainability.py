"""Explainability: which inputs the models rely on, read from reports/explain (phase 9)."""

import plotly.express as px
import streamlit as st

from dashboard import loaders

st.title("Explainability")
st.caption(
    "Two methods, computed by `python -m pipeline.explain` on the validation engines. For "
    "XGBoost, SHAP values share each prediction out between the inputs. For the LSTM, "
    "permutation importance shuffles one sensor and measures how much the error grows."
)

folder = loaders.path("reports", "explain")
files = {
    name: folder / name
    for name in [
        "xgboost_shap_importance.csv",
        "xgboost_shap_beeswarm.png",
        "lstm_permutation_importance.csv",
        "sensor_ranking.csv",
        "one_prediction.json",
        "xgboost_shap_one_prediction.png",
    ]
}
loaders.require(list(files.values()), "dvc repro")

st.subheader("XGBoost: global importance (SHAP)")
st.caption(
    "The mean size of each input's SHAP value over 2,000 validation rows: how many cycles it "
    "moves the prediction, on average, in either direction. Each sensor gives three inputs: its "
    "value, its 30-cycle mean (_mean) and spread (_std). The 15 largest are shown."
)
shap_importance = loaders.read_csv(files["xgboost_shap_importance.csv"])
shap_importance.columns = ["input", "mean_abs_shap"]
st.plotly_chart(
    px.bar(
        shap_importance.head(15).iloc[::-1],  # largest at the top
        x="mean_abs_shap",
        y="input",
        orientation="h",
        labels={"mean_abs_shap": "mean |SHAP value| (cycles)", "input": ""},
    ),
    alt="The 15 inputs with the largest mean absolute SHAP value for XGBoost",
)

st.subheader("XGBoost: every row, every input (beeswarm)")
st.caption(
    "One dot per validation row and per input. Left of 0: the input lowers the predicted life; "
    "right: it raises it. Colour: the input's own value (red high, blue low). A clean split of "
    "colours, like red on the left for s_3_mean, means a clear rule: a high value means less "
    "life left."
)
st.image(str(files["xgboost_shap_beeswarm.png"]), alt="SHAP beeswarm plot of XGBoost")

st.subheader("LSTM: permutation importance")
st.caption(
    "The LSTM reads 50-cycle windows, so SHAP does not apply as simply. Instead, one sensor's "
    "whole history is shuffled between windows, and the validation RMSE is measured again: the "
    "more it grows, the more the model needed that sensor. Mean of 5 shuffles."
)
permutation = loaders.read_csv(files["lstm_permutation_importance.csv"])
permutation.columns = ["sensor", "rmse_increase"]
st.plotly_chart(
    px.bar(
        permutation.iloc[::-1],
        x="rmse_increase",
        y="sensor",
        orientation="h",
        labels={"rmse_increase": "increase of the validation RMSE (cycles)", "sensor": ""},
    ),
    alt="Increase of the LSTM's validation RMSE when each sensor is shuffled",
)

st.subheader("Do the two models look at the same sensors?")
st.caption(
    "Each sensor's rank for each model (1 = most important). For XGBoost, a sensor's score adds "
    "up its three inputs. The rankings differ: the two models do not learn the same way, and "
    "correlated sensors can stand in for one another (s_9 and s_14, s_8 and s_13 move together)."
)
st.dataframe(
    loaders.read_csv(files["sensor_ranking.csv"]),
    hide_index=True,
    column_config={
        "sensor": "Sensor",
        "xgboost_shap": st.column_config.NumberColumn("XGBoost SHAP", format="%.2f"),
        "lstm_permutation": st.column_config.NumberColumn("LSTM permutation", format="%.2f"),
        "xgboost_rank": "XGBoost rank",
        "lstm_rank": "LSTM rank",
    },
)

st.subheader("One prediction explained")
one = loaders.read_json(files["one_prediction.json"])
st.caption(
    f"Validation engine {one['unit_id']} at cycle {one['cycle']}, {one['true_rul']} cycles "
    f"before failure. XGBoost starts from its average prediction ({one['base_value']:.1f} "
    f"cycles); each input then adds or removes cycles, and the arrows end on its prediction "
    f"({one['prediction']:.1f} cycles)."
)
st.image(
    str(files["xgboost_shap_one_prediction.png"]),
    alt=f"SHAP waterfall of one prediction for engine {one['unit_id']}",
)
