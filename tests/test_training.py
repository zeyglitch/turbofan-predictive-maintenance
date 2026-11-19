import numpy as np
import pytest
import torch

from turbofan.metrics import rmse
from turbofan.networks import LSTMRegressor
from turbofan.training import predict_network, set_seed, train_network

CAP = 125
PARAMS = {"batch_size": 32, "learning_rate": 0.01, "max_epochs": 5, "patience": 10}


def synthetic_windows(n: int, seed: int):
    """n windows of 30 cycles and 3 sensors; the target is the mean of sensor 0, times the cap."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 30, 3)).astype(np.float32)
    X[:, :, 0] += rng.uniform(-0.5, 0.5, size=(n, 1))  # each window gets its own level
    y = np.clip(X[:, :, 0].mean(axis=1), 0, 1) * CAP
    return X, y


def small_lstm() -> LSTMRegressor:
    return LSTMRegressor(n_sensors=3, hidden_size=8, num_layers=1, dropout=0.0)


def test_same_seed_gives_same_initial_weights():
    set_seed(7)
    first = small_lstm()
    set_seed(7)
    second = small_lstm()
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(a, b)


def test_loss_goes_down_on_an_easy_problem():
    X, y = synthetic_windows(200, seed=0)
    X_val, y_val = synthetic_windows(50, seed=1)
    set_seed(0)
    history = train_network(small_lstm(), X, y, X_val, y_val, PARAMS, CAP)
    assert len(history) == 5
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_model_keeps_the_weights_of_its_best_epoch():
    X, y = synthetic_windows(200, seed=0)
    X_val, y_val = synthetic_windows(50, seed=1)
    set_seed(0)
    model = small_lstm()
    history = train_network(model, X, y, X_val, y_val, PARAMS, CAP)
    best = min(row["val_rmse"] for row in history)
    assert rmse(y_val, predict_network(model, X_val, CAP)) == pytest.approx(best)


def test_training_stops_early_when_validation_never_improves():
    X, y = synthetic_windows(200, seed=0)
    X_val, _ = synthetic_windows(50, seed=1)
    noise = np.random.default_rng(2).uniform(0, CAP, size=50)  # nothing to learn on validation
    set_seed(0)
    params = PARAMS | {"max_epochs": 30, "patience": 1}
    history = train_network(small_lstm(), X, y, X_val, noise, params, CAP)
    assert len(history) < 30


def test_on_epoch_receives_every_row_of_the_history():
    X, y = synthetic_windows(64, seed=0)
    seen = []
    history = train_network(small_lstm(), X, y, X, y, PARAMS, CAP, on_epoch=seen.append)
    assert seen == history


def test_predictions_are_bounded_to_0_and_the_cap():
    model = small_lstm()
    X, _ = synthetic_windows(10, seed=0)
    with torch.no_grad():
        model.head.bias.fill_(10.0)  # raw output far above 1, i.e. far above the cap
    assert (predict_network(model, X, CAP) == CAP).all()
    with torch.no_grad():
        model.head.bias.fill_(-10.0)
    assert (predict_network(model, X, CAP) == 0).all()
