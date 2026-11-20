"""Train the LSTM and the CNN on windows of consecutive cycles, track every epoch, save them."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from turbofan.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.metrics import evaluate_predictions, rmse
from turbofan.networks import build_network
from turbofan.plots import plot_predictions
from turbofan.tracking import log_figure, log_metrics, log_model, start_run
from turbofan.training import predict_network, set_seed, train_network
from turbofan.windows import make_windows

NETWORKS = ["lstm", "cnn"]


def fit_network(
    params: dict,
    train,
    val,
    model_type: str,
    model_params: dict,
    length: int,
    run_name: str,
    tags=None,
):
    """Train one LSTM or CNN on windows of `length` cycles and record the run in MLflow.

    Returns the model (with the weights of its best epoch), its validation metrics and the
    history of the epochs. Also used by grid.py, with other window lengths and sizes.
    """
    sensors = params["prepare"]["sensors"]
    cap = params["prepare"]["rul_cap"]
    X_train, y_train, _ = make_windows(train, sensors, length)
    # one validation window per row: every window length, and the classic models, are scored
    # on the same rows
    X_val, y_val, _ = make_windows(val, sensors, length, every_cycle=True)
    print(f"{run_name}: {len(X_train)} training windows, {len(X_val)} validation windows")

    settings = (
        model_params
        | params["training"]
        | {
            "subset": params["data"]["subset"],
            "rul_cap": cap,
            "window_length": length,
            "split_seed": params["split"]["seed"],
        }
    )
    # same seed, same initial weights and same order of batches: the run can be repeated
    set_seed(params["training"]["seed"])
    model = build_network(model_type, len(sensors), model_params)

    def on_epoch(row: dict) -> None:
        # step=epoch: MLflow draws one point per epoch, which gives the learning curve.
        # "epoch_" names: val_rmse alone is the final score, comparable with the classic models
        log_metrics(
            {"epoch_train_loss": row["train_loss"], "epoch_val_rmse": row["val_rmse"]},
            step=row["epoch"],
        )
        print(
            f"  epoch {row['epoch']:>2}  train_loss {row['train_loss']:.4f}  "
            f"val_rmse {row['val_rmse']:.2f}"
        )

    with start_run(run_name, settings, tags):
        history = train_network(
            model, X_train, y_train, X_val, y_val, params["training"], cap, on_epoch
        )
        val_pred = predict_network(model, X_val, cap)
        metrics = evaluate_predictions(y_val, val_pred, cap, params["alert"]["threshold"])
        best_epoch = min(history, key=lambda row: row["val_rmse"])["epoch"]
        extra = {
            "train_rmse": rmse(y_train, predict_network(model, X_train, cap)),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
        }
        log_metrics({f"val_{key}": value for key, value in metrics.items()} | extra)
        figure = plot_predictions(y_val, val_pred, f"{run_name}, validation", cap)
        log_figure(figure, "pred_vs_true_val.png")
        plt.close(figure)
        log_model(model, "pytorch", input_example=X_val[:5])
    return model, metrics | extra, history


def main() -> None:
    params = load_params()
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    length = params["windows"]["length"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results, curves = {}, []
    for name in NETWORKS:
        model, metrics, history = fit_network(params, train, val, name, params[name], length, name)
        print(
            f"{name}: best epoch {metrics['best_epoch']} of {metrics['epochs_run']}, "
            f"train rmse {metrics['train_rmse']:.2f}, val rmse {metrics['rmse']:.2f}, "
            f"val mae {metrics['mae']:.2f}, recall {metrics['recall']:.2f}"
        )
        # only the weights: the architecture is rebuilt from params.yaml when the model is loaded
        torch.save(model.cpu().state_dict(), MODELS_DIR / f"{name}.pt")
        results[name] = metrics
        curves.append(pd.DataFrame(history).assign(model=name))

    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)  # creates reports/ as well
    # one row per epoch and network; lineterminator="\n" for the same reason as newline="\n"
    pd.concat(curves).to_csv(REPORTS_DIR / "training_curves.csv", index=False, lineterminator="\n")
    # newline="\n": the same bytes on Windows and Linux, so the fingerprint in dvc.lock matches
    (metrics_dir / "val_deep.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"wrote lstm.pt and cnn.pt to {MODELS_DIR}, training_curves.csv and val_deep.json")


if __name__ == "__main__":
    main()
