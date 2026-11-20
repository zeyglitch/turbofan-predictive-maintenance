import pytest
import torch

from turbofan.networks import CNNRegressor, LSTMRegressor, build_network


def test_lstm_returns_one_value_per_window():
    model = LSTMRegressor(n_sensors=14, hidden_size=64, num_layers=2, dropout=0.2)
    output = model(torch.zeros(4, 30, 14))
    assert output.shape == (4,)


def test_lstm_accepts_any_window_length():
    model = LSTMRegressor(n_sensors=14, hidden_size=8, num_layers=1, dropout=0.2)
    assert model(torch.zeros(2, 50, 14)).shape == (2,)


def test_cnn_returns_one_value_per_window():
    model = CNNRegressor(n_sensors=14, channels=[32, 64, 64], kernel_size=5)
    assert model(torch.zeros(4, 30, 14)).shape == (4,)


def test_cnn_accepts_any_window_length():
    model = CNNRegressor(n_sensors=14, channels=[8], kernel_size=5)
    assert model(torch.zeros(2, 50, 14)).shape == (2,)


def test_build_network_picks_the_architecture():
    assert isinstance(build_network("cnn", 14, {"channels": [8], "kernel_size": 3}), CNNRegressor)
    lstm_params = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    assert isinstance(build_network("lstm", 14, lstm_params), LSTMRegressor)
    with pytest.raises(ValueError, match="unknown network"):
        build_network("gru", 14, {})
