"""Retraining flow: drift report -> decision -> challenger -> comparison -> promotion -> report.

Run it after a simulation (the drift is read from monitoring/predictions.jsonl):
    python -m flows.retrain_flow
Prefect runs each task in order, records the run and retries a failing task. Without a Prefect
server, it starts a temporary one for the duration of the run. The challenger is the same model
type with the same settings as the champion, trained on FD001 + FD003; it replaces
models/champion only if the written rule of turbofan/retraining.py says so.
"""

import json
import shutil
from datetime import datetime, timezone

import joblib
import pandas as pd
from prefect import flow, task

from pipeline.monitor import load_reference
from pipeline.prepare import prepare
from pipeline.register import MODEL_NAME
from pipeline.train_deep import fit_network
from turbofan.bundle import load_bundle, save_bundle
from turbofan.config import MODELS_DIR, REPORTS_DIR, ROOT, load_params
from turbofan.conformal import conformal_quantile
from turbofan.data import load_raw, load_test_rul
from turbofan.labels import add_rul
from turbofan.metrics import rmse
from turbofan.monitoring import MONITORING_DIR, build_drift_report, load_prediction_log
from turbofan.predict import predict_engine
from turbofan.preprocessing import scale_sensors
from turbofan.retraining import decide
from turbofan.subsets import UNIT_OFFSET, load_subsets
from turbofan.tracking import git_state, last_run_id, register_champion, set_alias
from turbofan.training import predict_network
from turbofan.windows import make_windows

NETWORKS = ["lstm", "cnn"]
CHALLENGER_DATA = ROOT / "data" / "challenger"
CHALLENGER_DIR = MODELS_DIR / "challenger"
CHAMPION_DIR = MODELS_DIR / "champion"
REPORT_DIR = REPORTS_DIR / "retraining"

# the rule itself stays a plain function of turbofan/, tested without Prefect; wrapped here so
# that Prefect records it as a step of the run, like the others
decide_task = task(decide, name="decide")


@task(retries=2, retry_delay_seconds=5)
def check_drift(params: dict) -> dict:
    """The drift report on everything the API has logged, as pipeline/monitor.py computes it.

    retries: the log may be read while a simulation is still writing its last line.
    """
    settings = params["monitoring"]
    log = load_prediction_log(MONITORING_DIR / "predictions.jsonl")
    report = build_drift_report(
        load_reference(settings), log, params["prepare"]["sensors"], settings["drift_threshold"]
    )
    print(f"drift share {report['drift_share']:.2f} on {len(log)} logged predictions")
    return report | {"n_current": len(log)}


@task
def train_challenger(params: dict, champion: dict) -> dict:
    """Same model type and settings as the champion, trained on FD001 + FD003.

    Returns the challenger as an in-memory bundle {"model", "scaler", "metadata"}: the same
    shape as load_bundle, so predict_engine scores it exactly like the served champion. Also
    "val_units": its validation engines, on which the promotion is decided.
    Only networks are handled: the champion of this project is the LSTM.
    """
    model_type = champion["model_type"]
    if model_type not in NETWORKS:
        raise NotImplementedError(f"retraining a {model_type} champion is not written yet")
    subsets = [params["data"]["subset"], params["retraining"]["extra_subset"]]
    name = "+".join(subsets)

    # the same preparation as the DVC stage, on both subsets, in folders of its own
    processed = CHALLENGER_DATA / "processed"
    scaler_path = CHALLENGER_DATA / "scaler.joblib"
    prepared = prepare(subsets, params, processed, scaler_path, CHALLENGER_DATA / "prepare.json")
    train, val, calib = (
        pd.read_parquet(processed / f"{p}.parquet") for p in ("train", "val", "calib")
    )

    # the champion's settings, not params.yaml: the challenger differs by its data only
    length = champion["window_length"]
    run_params = params | {"data": params["data"] | {"subset": name}}
    model, metrics, _ = fit_network(
        run_params,
        train,
        val,
        model_type,
        champion["model_params"],
        length,
        f"challenger_{model_type}",
        tags={"alias": "challenger"},
    )
    run_id = last_run_id()
    version = register_champion(run_id, MODEL_NAME, "challenger")
    print(f"registered {MODEL_NAME} version {version} with alias challenger")

    # its 90 % interval, calibrated on the calibration engines of both subsets
    sensors, cap = champion["sensors"], champion["rul_cap"]
    X_calib, y_calib, _ = make_windows(calib, sensors, length, every_cycle=True)
    calib_pred = predict_network(model, X_calib, cap)
    q = conformal_quantile(y_calib, calib_pred, params["conformal"]["coverage"])

    kept = ("model_params", "sensors", "window_length", "rolling_window", "rul_cap")
    metadata = {key: champion[key] for key in kept} | {
        "model_type": model_type,
        "subsets": subsets,
        "conformal_q": q,
        "alert_threshold": params["alert"]["threshold"],
        "val_rmse": metrics["rmse"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_state()["git_commit"],
        "mlflow_run_id": run_id,
        "registry_version": version,
    }
    print(f"challenger: val rmse {metrics['rmse']:.2f} on {name}, interval q {q:.1f}")
    return {
        "model": model,
        "scaler": joblib.load(scaler_path),
        "metadata": metadata,
        "val_units": prepared["parts"]["val"]["units"],
    }


def score_on_validation_engines(bundle: dict, engines: pd.DataFrame, subsets: list[str]) -> dict:
    """RMSE on every cycle of the validation engines, per subset and on all of them together.

    engines: the raw rows of the validation engines (training files, followed to failure), with
    rul_clipped. Neither model was trained on them. The FD001 ones are the validation engines of
    the champion too (same seed, same share): both chose their best epoch on them; the
    challenger also chose it on the FD003 ones, which leans slightly in its favour.
    """
    metadata = bundle["metadata"]
    sensors, cap = metadata["sensors"], metadata["rul_cap"]
    # each model with its own scaler, as predict_engine does
    df = scale_sensors(engines, bundle["scaler"], sensors)
    X, y, units = make_windows(df, sensors, metadata["window_length"], every_cycle=True)
    pred = predict_network(bundle["model"], X, cap)
    scores = {
        subset: rmse(y[units // UNIT_OFFSET == position], pred[units // UNIT_OFFSET == position])
        for position, subset in enumerate(subsets)
    }
    scores["all"] = rmse(y, pred)
    return scores


def score_on_test_engines(bundle: dict, subsets: list[str]) -> dict:
    """RMSE at the last cycle of each test engine, per subset and on all of them together.

    Every engine goes through predict_engine, the function behind the API: the models are
    scored on exactly what they would answer in service.
    """
    cap = bundle["metadata"]["rul_cap"]
    y_true, y_pred, scores = [], [], {}
    for subset in subsets:
        test = load_raw(subset, "test")
        truth = load_test_rul(subset).clip(upper=cap)
        pred = [predict_engine(bundle, engine)["rul"] for _, engine in test.groupby("unit_id")]
        scores[subset] = rmse(truth, pred)
        y_true += truth.tolist()
        y_pred += pred
    scores["all"] = rmse(y_true, y_pred)
    return scores


@task
def evaluate_both(champion: dict, challenger: dict, subsets: list[str], cap: int) -> dict:
    """Both models on the validation engines (they decide) and on the test engines (reported).

    Validation: the 30 validation engines of FD001 and FD003, every cycle. Test: the 200 test
    engines, last cycle, as in the pipeline's evaluation; read after the decision is possible,
    never used by it.
    """
    train_rows = add_rul(load_subsets(subsets, "train"), cap)
    engines = train_rows[train_rows["unit_id"].isin(challenger["val_units"])]
    scores = {
        "validation": {
            "champion": score_on_validation_engines(champion, engines, subsets),
            "challenger": score_on_validation_engines(challenger, engines, subsets),
        },
        "test": {
            "champion": score_on_test_engines(champion, subsets),
            "challenger": score_on_test_engines(challenger, subsets),
        },
    }
    for part, models in scores.items():
        for who, own in models.items():
            print(f"{part} {who}: " + ", ".join(f"{key} {value:.2f}" for key, value in own.items()))
    return scores


@task
def promote(challenger: dict, scores: dict, decision: dict) -> None:
    """Save the challenger in models/challenger; if promoted, it becomes models/champion."""
    metadata = challenger["metadata"] | {"test_rmse": scores["test"]["challenger"]["all"]}
    shutil.rmtree(CHALLENGER_DIR, ignore_errors=True)
    save_bundle(
        CHALLENGER_DIR,
        metadata["model_type"],
        challenger["model"],
        challenger["scaler"],
        metadata,
    )
    if not decision["promote"]:
        print(f"champion kept; the challenger stays in {CHALLENGER_DIR}")
        return
    # the delivered folder is replaced whole, and the registry label moves to the new version
    shutil.rmtree(CHAMPION_DIR)
    shutil.copytree(CHALLENGER_DIR, CHAMPION_DIR)
    set_alias(MODEL_NAME, "champion", metadata["registry_version"])
    print(
        f"challenger promoted: {CHAMPION_DIR} replaced, champion = v{metadata['registry_version']}"
    )


@task
def write_report(report: dict) -> str:
    """reports/retraining/<date>.json: trigger, drift, scores, decision and reason."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # 2026-11-22T154144Z.json: no colon in the file name, Windows refuses it
    started = datetime.fromisoformat(report["started_at"])
    path = REPORT_DIR / f"{started.strftime('%Y-%m-%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return str(path)


@flow(name="retrain-turbofan", log_prints=True)
def retrain_flow() -> dict:
    params = load_params()
    settings = params["monitoring"] | params["retraining"]
    subsets = [params["data"]["subset"], params["retraining"]["extra_subset"]]
    champion = load_bundle(CHAMPION_DIR)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": "drift report on monitoring/predictions.jsonl",
        "champion_version": champion["metadata"]["registry_version"],
    }

    drift = check_drift(params)
    report |= {
        "drift_share": drift["drift_share"],
        "drifted_sensors": drift["drifted_sensors"],
        "n_logged": drift["n_current"],
    }
    if drift["drift_share"] < settings["drift_share_threshold"]:
        # nothing to retrain: the decision says so and the report keeps a trace of it
        report["decision"] = decide_task(drift["drift_share"], None, None, settings)
        write_report(report)
        return report

    challenger = train_challenger(params, champion["metadata"])
    scores = evaluate_both(champion, challenger, subsets, params["prepare"]["rul_cap"])
    # the validation engines decide; the test scores are only reported
    validation = scores["validation"]
    decision = decide_task(
        drift["drift_share"], validation["champion"], validation["challenger"], settings
    )
    print(f"decision: {'promote' if decision['promote'] else 'keep the champion'}")
    print(f"reason: {decision['reason']}")
    promote(challenger, scores, decision)
    report |= {
        "challenger_version": challenger["metadata"]["registry_version"],
        "challenger_val_rmse": challenger["metadata"]["val_rmse"],
        "scores": scores,
        "decision": decision,
    }
    write_report(report)
    return report


if __name__ == "__main__":
    retrain_flow()
