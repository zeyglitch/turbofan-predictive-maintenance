import mlflow

from turbofan import tracking
from turbofan.tracking import EXPERIMENT_NAME, git_state, log_metrics, start_run, tracking_uri


def test_run_records_params_metrics_and_git_state(tmp_path, monkeypatch):
    # a throwaway database, so the test never writes into the project's mlflow.db
    uri = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    with start_run("unit-test", {"alpha": 1.0, "rul_cap": 125}, tags={"purpose": "test"}) as run:
        log_metrics({"val_rmse": 20.5})

    stored = mlflow.get_run(run.info.run_id)
    assert tracking_uri() == uri
    assert mlflow.get_experiment(stored.info.experiment_id).name == EXPERIMENT_NAME
    # MLflow stores every param as text
    assert stored.data.params == {"alpha": "1.0", "rul_cap": "125"}
    assert stored.data.metrics == {"val_rmse": 20.5}
    assert stored.data.tags["purpose"] == "test"
    assert stored.data.tags["git_commit"] == git_state()["git_commit"]
    assert stored.data.tags["git_dirty"] in {"true", "false", "unknown"}


def test_git_state_is_unknown_without_git(monkeypatch):
    # pretend git is not installed, as in a slim container image
    monkeypatch.setattr(tracking.shutil, "which", lambda name: None)

    assert git_state() == {"git_commit": "unknown", "git_dirty": "unknown"}


def test_git_state_is_unknown_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "ROOT", tmp_path)

    assert git_state() == {"git_commit": "unknown", "git_dirty": "unknown"}
