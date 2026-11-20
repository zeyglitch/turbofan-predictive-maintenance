import mlflow
import torch

from turbofan import tracking
from turbofan.networks import LSTMRegressor
from turbofan.tracking import (
    EXPERIMENT_NAME,
    git_state,
    log_metrics,
    log_model,
    start_run,
    tracking_uri,
)


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


def test_logged_lstm_reloads_and_predicts_the_same_on_any_batch_size(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(tracking, "ROOT", tmp_path)  # attached files go to tmp_path/mlartifacts
    model = LSTMRegressor(n_sensors=3, hidden_size=4, num_layers=2, dropout=0.2).eval()
    X = torch.rand(7, 30, 3)

    with start_run("unit-test", {}):
        log_model(model, "pytorch", input_example=X[:5].numpy())

    logged = mlflow.search_logged_models(output_format="list")[0]
    reloaded = mlflow.pytorch.load_model(logged.model_uri)
    with torch.no_grad():
        # 7 windows although the example had 5: the saved graph must not fix the batch size
        assert torch.allclose(reloaded(X), model(X))


def test_git_state_is_unknown_without_git(monkeypatch):
    # pretend git is not installed, as in a slim container image
    monkeypatch.setattr(tracking.shutil, "which", lambda name: None)

    assert git_state() == {"git_commit": "unknown", "git_dirty": "unknown"}


def test_git_state_is_unknown_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "ROOT", tmp_path)

    assert git_state() == {"git_commit": "unknown", "git_dirty": "unknown"}
