import numpy as np

from turbofan.classic_models import (
    predict,
    train_baseline,
    train_random_forest,
    train_ridge,
    train_xgboost,
)
from turbofan.metrics import rmse


def make_problem(n_rows: int, seed: int):
    """y depends on the first of two columns only, plus a little noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 2))
    y = 3 * X[:, 0] + rng.normal(scale=0.1, size=n_rows)
    return X, y


def test_baseline_is_the_training_mean():
    assert train_baseline([10, 20, 30]) == 20.0


def test_ridge_beats_the_baseline_on_a_linear_problem():
    X, y = make_problem(200, seed=0)

    model = train_ridge(X, y, {"alpha": 1.0})
    baseline = train_baseline(y)

    assert rmse(y, model.predict(X)) < rmse(y, np.full(200, baseline))
    assert abs(model.coef_[0] - 3) < 0.1
    assert abs(model.coef_[1]) < 0.1


def test_random_forest_predicts_every_row():
    X, y = make_problem(100, seed=1)

    forest = train_random_forest(X, y, {"n_estimators": 10, "max_depth": 3, "random_state": 0})
    predictions = predict(forest, X)

    assert predictions.shape == (100,)
    assert not np.isnan(predictions).any()


def test_random_forest_predictions_are_identical_from_one_call_to_the_next():
    X, y = make_problem(2000, seed=4)

    forest = train_random_forest(X, y, {"n_estimators": 50, "random_state": 0})
    calls = [predict(forest, X).tobytes() for _ in range(5)]

    # bytes, not approximate values: even the last digit must not move
    assert len(set(calls)) == 1


def test_xgboost_stops_early_and_predicts_every_row():
    X, y = make_problem(200, seed=2)
    X_val, y_val = make_problem(50, seed=3)
    params = {
        "n_estimators": 500,
        "learning_rate": 0.3,
        "max_depth": 2,
        "early_stopping_rounds": 5,
        "random_state": 0,
    }

    booster = train_xgboost(X, y, X_val, y_val, params)
    predictions = predict(booster, X_val)

    # a simple problem is learnt long before 500 trees, so early stopping kicks in
    assert booster.best_iteration < 499
    assert predictions.shape == (50,)
    assert rmse(y_val, predictions) < rmse(y_val, np.full(50, train_baseline(y)))


def test_predict_turns_the_baseline_number_into_one_value_per_row():
    assert predict(20.0, np.zeros((3, 2))).tolist() == [20.0, 20.0, 20.0]
