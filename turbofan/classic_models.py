"""Models that read one row of features at a time."""

import numpy as np
from sklearn.linear_model import Ridge


def train_baseline(y_train) -> float:
    """Return the mean target of the training rows: the constant every engine will be given."""
    return float(np.mean(y_train))


def train_ridge(X, y, params: dict) -> Ridge:
    """Fit a linear regression whose weights are pulled towards 0 with strength params["alpha"]."""
    return Ridge(alpha=params["alpha"]).fit(X, y)
