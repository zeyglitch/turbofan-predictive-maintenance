import numpy as np

from turbofan.classic_models import train_baseline, train_ridge
from turbofan.metrics import rmse


def test_baseline_is_the_training_mean():
    assert train_baseline([10, 20, 30]) == 20.0


def test_ridge_beats_the_baseline_on_a_linear_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    # y depends on the first column only, plus a little noise
    y = 3 * X[:, 0] + rng.normal(scale=0.1, size=200)

    model = train_ridge(X, y, {"alpha": 1.0})
    baseline = train_baseline(y)

    assert rmse(y, model.predict(X)) < rmse(y, np.full(200, baseline))
    assert abs(model.coef_[0] - 3) < 0.1
    assert abs(model.coef_[1]) < 0.1
