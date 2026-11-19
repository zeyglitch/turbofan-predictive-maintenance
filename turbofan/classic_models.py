"""Models that read one row of features at a time.

Each params dictionary comes from params.yaml and uses the library's own argument names.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def train_baseline(y_train) -> float:
    """Return the mean target of the training rows: the constant every engine will be given."""
    return float(np.mean(y_train))


def train_ridge(X, y, params: dict) -> Ridge:
    """Fit a linear regression whose weights are pulled towards 0 with strength params["alpha"]."""
    return Ridge(**params).fit(X, y)


def train_random_forest(X, y, params: dict) -> RandomForestRegressor:
    """Fit many decision trees, each on a random sample of rows, and average their answers."""
    # n_jobs=-1 trains the trees on every processor core: 5 s instead of 66 s, same trees
    forest = RandomForestRegressor(**params, n_jobs=-1).fit(X, y)
    # predicting on several cores adds the tree outputs in a varying order, which changes the last
    # digits from one call to the next; on one core the predictions are identical every time
    return forest.set_params(n_jobs=1)


def train_xgboost(X, y, X_val, y_val, params: dict) -> XGBRegressor:
    """Add trees one after the other, each correcting the previous ones.

    Training stops when the validation error has not improved for early_stopping_rounds trees;
    predict then uses the best number of trees.
    """
    return XGBRegressor(**params).fit(X, y, eval_set=[(X_val, y_val)], verbose=False)


def predict(model, X) -> np.ndarray:
    """Return the predictions of any model above; the baseline is a number, not an object."""
    if isinstance(model, float):
        return np.full(len(X), model)
    return model.predict(X)
