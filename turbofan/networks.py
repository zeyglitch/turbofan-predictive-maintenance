"""Neural networks that read a window of consecutive cycles and return one remaining life."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM layers, then a linear layer applied to the output of the last time step."""

    def __init__(self, n_sensors: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # inputs are (batch, time, sensors); without this flag LSTM expects (time, batch, …)
            batch_first=True,
            # dropout is applied between stacked layers, so a single layer has none to apply
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, time, sensors) -> (batch,)"""
        outputs, _ = self.lstm(x)  # (batch, time, hidden_size): one output per time step
        last = outputs[:, -1, :]  # (batch, hidden_size): the summary after the last cycle
        return self.head(last).squeeze(-1)  # (batch, 1) -> (batch,)


class CNNRegressor(nn.Module):
    """1D convolutions along time, an average over time, then a linear layer.

    dropout > 0 switches off that share of the filter outputs at random after each convolution,
    during training only: the network cannot lean on a few filters that fit the training engines.
    """

    def __init__(self, n_sensors: int, channels: list[int], kernel_size: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_channels = n_sensors
        for out_channels in channels:
            # padding keeps the length: a kernel of 5 sees 2 cycles on each side
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                # only when used: without it the layers keep their numbers, and saved weights load
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.convolutions = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, time, sensors) -> (batch,)"""
        x = x.transpose(1, 2)  # Conv1d expects (batch, channels, time): the sensors are channels
        x = self.convolutions(x)  # (batch, channels[-1], time)
        x = x.mean(dim=2)  # (batch, channels[-1]): the average of each filter over the window
        return self.head(x).squeeze(-1)  # (batch, 1) -> (batch,)


def build_network(model_type: str, n_sensors: int, model_params: dict) -> nn.Module:
    """Create an untrained LSTM or CNN from its section of params.yaml."""
    if model_type == "lstm":
        return LSTMRegressor(n_sensors=n_sensors, **model_params)
    if model_type == "cnn":
        return CNNRegressor(n_sensors=n_sensors, **model_params)
    raise ValueError(f"unknown network {model_type}, expected lstm or cnn")
