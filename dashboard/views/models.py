"""Models: the six models compared, then one model in detail."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import loaders

NETWORK_COLOURS = {"lstm": "#1f77b4", "cnn": "#ff7f0e"}  # blue and orange, easy to tell apart

st.title("Models")
st.caption(
    "Six models predict the remaining life: a constant baseline, Ridge, a random forest, XGBoost, "
    "an LSTM and a 1D CNN. The champion is chosen on the validation engines; the test engines "
    "are only used to report the final scores below, never to choose."
)

subsets_file = loaders.path("reports", "subsets.json")
if subsets_file.exists():
    st.subheader("Across the four subsets")
    runs = loaders.read_json(subsets_file)
    labels = {
        run["name"]: f"{run['subset']}"
        + (" (scaled within regimes)" if run["normalize_by_regime"] else "")
        + (" — the pipeline" if run["name"] == "pipeline" else "")
        for run in runs
    }
    st.caption(
        "The same chain, with the same settings, run on each subset (DVC experiments, collected "
        "by `python -m pipeline.collect_subsets`). FD002 and FD004 fly at six regimes and are "
        "scaled within each regime; FD003 and FD004 have two fault modes. Test RMSE at the last "
        "cycle of each test engine; the champion of each subset would be chosen on validation."
    )
    comparison = pd.DataFrame(
        [
            {"run": labels[run["name"]], "model": model, "test RMSE": scores["test_rmse"]}
            for run in runs
            for model, scores in run["models"].items()
            if model != "baseline"  # around 40 cycles everywhere, it would flatten the chart
        ]
    )
    st.plotly_chart(
        px.bar(
            comparison,
            x="model",
            y="test RMSE",
            color="run",
            barmode="group",
            # five clearly different colours: the theme's default gives two close blues
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"],
        ),
        alt="Test RMSE of each model on each subset",
    )
    chosen = st.segmented_control(
        "Scores of one run",
        list(labels),
        format_func=labels.get,
        default="pipeline",
        required=True,
    )
    run = next(run for run in runs if run["name"] == chosen)
    st.caption(
        f"{run['n_train_engines']} engines to train, validation and calibration engines apart, "
        f"{run['n_test_engines']} test engines."
    )
    st.dataframe(
        pd.DataFrame(run["models"]).T.rename_axis("model").reset_index(),
        hide_index=True,
        column_config={
            "val_rmse": st.column_config.NumberColumn("val RMSE", format="%.1f"),
            "test_rmse": st.column_config.NumberColumn("test RMSE", format="%.1f"),
            "test_mae": st.column_config.NumberColumn("test MAE", format="%.1f"),
            "test_nasa_score": st.column_config.NumberColumn("test NASA score", format="%.0f"),
            "coverage": st.column_config.NumberColumn("interval coverage", format="%.2f"),
        },
    )
    st.caption("Everything below is the pipeline on FD001.")

val_classic_file = loaders.path("reports", "metrics", "val_classic.json")
val_deep_file = loaders.path("reports", "metrics", "val_deep.json")
test_file = loaders.path("reports", "metrics", "test.json")
champion_file = loaders.path("reports", "champion.json")
predictions_file = loaders.path("reports", "predictions_test.parquet")
curves_file = loaders.path("reports", "training_curves.csv")
loaders.require(
    [val_classic_file, val_deep_file, test_file, champion_file, predictions_file, curves_file],
    "dvc repro",
)

validation = loaders.read_json(val_classic_file) | loaders.read_json(val_deep_file)
test = loaders.read_json(test_file)
champion = loaders.read_json(champion_file)["champion"]
cap = loaders.params()["prepare"]["rul_cap"]
threshold = loaders.params()["alert"]["threshold"]

st.subheader("Scores")
st.caption(
    "RMSE and MAE are in cycles: the typical size of the error. The NASA score punishes late "
    "predictions (saying an engine has more life than it has) harder than early ones; lower is "
    "better. Validation scores use every cycle of 15 engines (2,834 rows), test scores the last "
    "cycle of 100 engines: compare models within a column, not across columns. Coverage: how "
    "often the true value falls inside the 90 % interval."
)
scores = pd.DataFrame(
    {
        "model": list(test),
        "val RMSE": [validation[name]["rmse"] for name in test],
        "val MAE": [validation[name]["mae"] for name in test],
        "test RMSE": [test[name]["rmse"] for name in test],
        "test MAE": [test[name]["mae"] for name in test],
        "test NASA score": [test[name]["nasa_score"] for name in test],
        "interval coverage": [test[name]["coverage"] for name in test],
    }
)
scores["model"] = [f"{name} (champion)" if name == champion else name for name in scores["model"]]
st.dataframe(
    scores,
    hide_index=True,
    column_config={
        column: st.column_config.NumberColumn(format="%.1f")
        for column in ["val RMSE", "val MAE", "test RMSE", "test MAE", "test NASA score"]
    }
    | {"interval coverage": st.column_config.NumberColumn(format="%.2f")},
)

st.subheader("Learning curves of the networks")
st.caption(
    "One point per epoch (one pass over the training windows). Left: the training loss, on the "
    "target divided by the cap. Right: the validation RMSE, in cycles. Training stops when the "
    "validation RMSE has not improved for 8 epochs, and the weights of the best epoch are kept. "
    "A training loss that keeps falling while the validation RMSE rises is overfitting."
)
curves = loaders.read_csv(curves_file)
left, right = st.columns(2)
left.plotly_chart(
    px.line(
        curves,
        x="epoch",
        y="train_loss",
        color="model",
        markers=True,
        color_discrete_map=NETWORK_COLOURS,
    ),
    alt="Training loss of the LSTM and the CNN per epoch",
)
right.plotly_chart(
    px.line(
        curves,
        x="epoch",
        y="val_rmse",
        color="model",
        markers=True,
        color_discrete_map=NETWORK_COLOURS,
    ),
    alt="Validation RMSE of the LSTM and the CNN per epoch",
)
best = ", ".join(
    f"{name}: best epoch {validation[name]['best_epoch']} of {validation[name]['epochs_run']}"
    for name in curves["model"].unique()
)
st.caption(best)

st.subheader("One model in detail, on the test engines")
names = list(test)
model = st.segmented_control("Model", names, default=champion, required=True)
predictions = loaders.read_parquet(predictions_file)
chosen = predictions[predictions["model"] == model].assign(
    error=lambda df: df["rul_pred"] - df["rul_true"]
)

left, right = st.columns(2)
figure = px.scatter(
    chosen,
    x="rul_true",
    y="rul_pred",
    error_y=chosen["rul_high"] - chosen["rul_pred"],
    error_y_minus=chosen["rul_pred"] - chosen["rul_low"],
    hover_data=["unit_id"],
    labels={"rul_true": "True remaining life", "rul_pred": "Predicted remaining life"},
    title="Predicted vs true, 90 % interval",
)
figure.add_shape(type="line", x0=0, y0=0, x1=cap, y1=cap, line={"dash": "dot", "color": "grey"})
left.plotly_chart(figure, alt=f"Predicted against true remaining life for {model}")
right.plotly_chart(
    px.histogram(
        chosen,
        x="error",
        nbins=30,
        labels={"error": "Predicted minus true (cycles)"},
        title="Errors (above 0: late)",
    ),
    alt=f"Histogram of the errors of {model}",
)
st.caption(
    "One point per test engine, at its last known cycle. On the dotted line the prediction is "
    "exact; above it the model is late (it gives more life than there is), which is the "
    "dangerous side."
)

st.markdown(
    f"**The alert: an engine is flagged when its predicted life is {threshold} cycles or less.**"
)
counts = test[model]
matrix = pd.DataFrame(
    [[counts["tp"], counts["fn"]], [counts["fp"], counts["tn"]]],
    index=[f"truly ≤ {threshold} cycles", f"truly > {threshold} cycles"],
    columns=["alert raised", "no alert"],
)
left, right = st.columns([2, 1])
left.plotly_chart(
    px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", aspect="auto"),
    alt=f"Confusion matrix of the alert for {model}",
)
right.metric(
    "Precision",
    f"{counts['precision']:.2f}",
    help="Of the alerts raised, the share that were right.",
    border=True,
)
right.metric(
    "Recall",
    f"{counts['recall']:.2f}",
    help="Of the engines that truly needed an alert, the share that got one.",
    border=True,
)
st.caption(
    "A missed alert (top right: truly close to failure, but no alert) is the costly mistake; a "
    "false alarm (bottom left) only costs an early inspection."
)

st.subheader("Comparison with the literature")
st.caption(
    "Two well-known papers on FD001. Their protocols differ from this project (cap of 125 or "
    "130, window length, which test cycle is scored), so a comparison would only be indicative. "
    "Their figures have not been checked against the original articles yet, so none is shown "
    "here: a number is only written after its source has been read."
)
st.dataframe(
    pd.DataFrame(
        {
            "reference": [
                "Zheng, Ristovski, Farahat, Gupta (2017), IEEE ICPHM",
                "Li, Ding, Sun (2018), Reliability Engineering & System Safety 172",
            ],
            "model": ["LSTM", "Deep CNN"],
            "FD001 test RMSE": ["not verified yet", "not verified yet"],
        }
    ),
    hide_index=True,
)
