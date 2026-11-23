"""Overview: the key figures, the pipeline, and when each stage last ran."""

import streamlit as st

from dashboard import loaders

st.title("Overview")
st.caption(
    "Predicting how many flight cycles a turbofan engine has left, on NASA's simulated C-MAPSS "
    "data (subset FD001). This dashboard only reads the files written by the pipeline "
    "(`dvc repro`): it trains nothing and computes no model."
)

prepare_file = loaders.path("reports", "prepare.json")
champion_file = loaders.path("reports", "champion.json")
predictions_file = loaders.path("reports", "predictions_test.parquet")
loaders.require([prepare_file, champion_file, predictions_file], "dvc repro")

parts = loaders.read_json(prepare_file)["parts"]
champion = loaders.read_json(champion_file)
threshold = loaders.params()["alert"]["threshold"]
predictions = loaders.read_parquet(predictions_file)
champion_predictions = predictions[predictions["model"] == champion["champion"]]
n_alerts = int((champion_predictions["rul_pred"] <= threshold).sum())

with st.container(horizontal=True):
    st.metric(
        "Training engines",
        sum(parts[name]["n_engines"] for name in ("train", "val", "calib")),
        help=(
            f"{parts['train']['n_engines']} to train, {parts['val']['n_engines']} to choose "
            f"the model, {parts['calib']['n_engines']} to calibrate the intervals."
        ),
        border=True,
    )
    st.metric(
        "Test engines",
        parts["test"]["n_engines"],
        help="Kept aside, used only for the final evaluation.",
        border=True,
    )
    st.metric(
        "Champion model",
        f"{champion['champion'].upper()} v{champion['registry_version']}",
        help="The model with the lowest validation RMSE, registered in MLflow.",
        border=True,
    )
    st.metric(
        "Champion test RMSE",
        f"{champion['test_rmse']:.1f} cycles",
        help=(
            "Typical size of the error on the 100 test engines, at their last cycle, with the "
            "true remaining life capped at 125 like the training target. Against the true value, "
            "not capped, it is higher: see Models."
        ),
        border=True,
    )
    st.metric(
        "Test engines in alert",
        f"{n_alerts} / {len(champion_predictions)}",
        help=f"Engines whose predicted remaining life is {threshold} cycles or less.",
        border=True,
    )

# the retraining flow (phase 14) can replace models/champion without going through DVC
bundle_file = loaders.path("models", "champion", "metadata.json")
if bundle_file.exists():
    served = loaders.read_json(bundle_file)
    if served["registry_version"] != champion["registry_version"]:
        st.warning(
            f"The retraining flow promoted version {served['registry_version']} "
            f"({served['model_type'].upper()}, trained on {' + '.join(served['subsets'])}): it "
            "is the model the API serves now. The figures above are those of the pipeline's "
            "champion. See the Retraining page.",
            title="The champion was replaced by retraining",
            icon=":material/autorenew:",
        )

st.subheader("The pipeline")
st.caption(
    "Each box is a stage of `dvc.yaml`. An arrow means: this stage writes a file that the next "
    "one needs, so it runs first (shortcuts implied by a longer path are left out). `dvc repro` "
    "reruns only the stages whose inputs changed."
)
dvc_file = loaders.path("dvc.yaml")
loaders.require([dvc_file], "git checkout dvc.yaml")
stages = loaders.pipeline_stages()
edges = loaders.drop_shortcuts(loaders.pipeline_edges(stages))
diagram = "flowchart LR\n" + "\n".join(f"    {a} --> {b}" for a, b in edges)
st.mermaid_chart(diagram, alt="The DVC pipeline, from download to register and explain")

st.subheader("Stages")
st.caption(
    "When the outputs of each stage were last written; empty: not run yet. The dates do not "
    "always follow the order of the pipeline: when a stage runs again and writes exactly the "
    "same files, DVC sees that nothing changed downstream and does not rerun the next stages."
)
st.dataframe(
    loaders.stage_status(stages),
    hide_index=True,
    column_config={
        "stage": "Stage",
        "outputs": "Outputs",
        "last_written": st.column_config.DatetimeColumn(
            "Outputs last written (UTC)", format="YYYY-MM-DD HH:mm"
        ),
    },
)
