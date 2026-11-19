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
