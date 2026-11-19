"""Training loop for the networks, written out step by step, with early stopping."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from turbofan.metrics import rmse


def set_seed(seed: int) -> None:
    """Fix every random generator used here: weight initialisation, shuffling, dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """The graphics card when PyTorch can use one, the processor otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_network(
    model: nn.Module, X_train, y_train, X_val, y_val, params: dict, rul_cap: int, on_epoch=None
) -> list[dict]:
    """Train with Adam on the mean squared error, one epoch at a time.

    After each epoch the validation RMSE (in cycles) is measured. Training stops after
    params["patience"] epochs without improvement, and the model gets back its best weights.
    Returns one {epoch, train_loss, val_rmse} per epoch; on_epoch(row) is called after each one.
    """
    device = get_device()
    model.to(device)
    # targets divided by the cap: between 0 and 1, the gradients stay of a reasonable size
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(np.asarray(y_train) / rul_cap, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.MSELoss()

    history = []
    best_rmse, best_weights, epochs_without_progress = float("inf"), None, 0
    for epoch in range(1, params["max_epochs"] + 1):
        model.train()  # dropout on
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  # forget the gradients of the previous batch
            loss = loss_fn(model(X_batch), y_batch)  # forward pass, then the loss
            loss.backward()  # backpropagation: d(loss)/d(weight) for every weight
            optimizer.step()  # move every weight a small step against its gradient
            total_loss += loss.item() * len(X_batch)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(dataset),
            "val_rmse": rmse(y_val, predict_network(model, X_val, rul_cap)),
        }
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)

        if row["val_rmse"] < best_rmse:
            # deepcopy: state_dict() returns the live tensors, which the next steps would change
            best_rmse, best_weights = row["val_rmse"], copy.deepcopy(model.state_dict())
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
            if epochs_without_progress >= params["patience"]:
                break

    model.load_state_dict(best_weights)
    return history


def predict_network(model: nn.Module, X, rul_cap: int, batch_size: int = 4096) -> np.ndarray:
    """Remaining life in cycles for each window of X, bounded to [0, rul_cap]."""
    device = next(model.parameters()).device
    model.eval()  # dropout off: the same input always gives the same output
    predictions = []
    with torch.no_grad():  # no gradient to compute: faster, less memory
        for X_batch in torch.as_tensor(X, dtype=torch.float32).split(batch_size):
            predictions.append(model(X_batch.to(device)).cpu().numpy())
    # back from [0, 1] to cycles
    return np.clip(np.concatenate(predictions) * rul_cap, 0, rul_cap)
