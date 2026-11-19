import torch

from turbofan.networks import LSTMRegressor


def test_lstm_returns_one_value_per_window():
    model = LSTMRegressor(n_sensors=14, hidden_size=64, num_layers=2, dropout=0.2)
    output = model(torch.zeros(4, 30, 14))
    assert output.shape == (4,)


def test_lstm_accepts_any_window_length():
    model = LSTMRegressor(n_sensors=14, hidden_size=8, num_layers=1, dropout=0.2)
    assert model(torch.zeros(2, 50, 14)).shape == (2,)
