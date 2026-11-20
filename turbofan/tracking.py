"""The only module that talks to MLflow: where runs are stored and what each run records."""

import os
import shutil
import subprocess
from contextlib import contextmanager

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost

from turbofan.config import ROOT

EXPERIMENT_NAME = "turbofan-rul"
# MLflow saves scikit-learn models with skops, which refuses unknown types when loading them back.
# Tree is the node storage of our own random forest: we create these files, so we trust it.
TRUSTED_SKLEARN_TYPES = ["sklearn.tree._tree.Tree"]


def tracking_uri() -> str:
    """Return MLFLOW_TRACKING_URI if it is set, otherwise the mlflow.db file at the project root."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def git_state() -> dict:
    """Return the short commit hash and whether the working copy has uncommitted changes.

    Both are "unknown" when git is not installed or the folder is not a git repository
    (inside a container, for instance).
    """
    unknown = {"git_commit": "unknown", "git_dirty": "unknown"}
    if shutil.which("git") is None:
        return unknown
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    if commit.returncode != 0:
        return unknown
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    # any line in the short status means a file was changed since the last commit
    return {
        "git_commit": commit.stdout.strip(),
        "git_dirty": str(bool(status.stdout.strip())).lower(),
    }


def use_experiment() -> None:
    """Point MLflow at our store and experiment, creating the experiment the first time."""
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        # without an explicit location, attached files would go to ./mlruns of the current folder
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def start_run(run_name: str, params: dict, tags: dict | None = None):
    """Open an MLflow run that records params and git state; use it with a `with` block."""
    use_experiment()
    with mlflow.start_run(run_name=run_name, tags=git_state() | (tags or {})) as run:
        mlflow.log_params(params)
        yield run


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Record numbers in the active run; step numbers a curve (one point per epoch later on)."""
    mlflow.log_metrics(metrics, step=step)


def log_figure(figure, file_name: str) -> None:
    """Attach a matplotlib figure to the active run as an image."""
    mlflow.log_figure(figure, file_name)


def log_model(model, flavour: str, input_example) -> None:
    """Attach the trained model to the active run, in the format of its library."""
    if flavour == "sklearn":
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            skops_trusted_types=TRUSTED_SKLEARN_TYPES,
        )
    elif flavour == "xgboost":
        mlflow.xgboost.log_model(model, name="model", input_example=input_example)
    elif flavour == "pytorch":
        mlflow.pytorch.log_model(model, name="model", input_example=input_example)
    else:
        raise ValueError(f"unknown model flavour {flavour}, expected sklearn, xgboost or pytorch")
