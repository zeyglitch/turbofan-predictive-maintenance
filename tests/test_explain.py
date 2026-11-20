import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from turbofan.explain import (
    importance_per_sensor,
    permutation_importance,
    shap_importance,
    shap_values,
)


def toy_problem(n: int = 300):
    """y = 10 * a + noise; b is pure noise and plays no part in y."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 10 * X["a"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_shap_values_add_up_to_the_prediction():
    X, y = toy_problem()
    model = XGBRegressor(n_estimators=50, max_depth=3).fit(X, y)
    explanation = shap_values(model, X)
    total = explanation.values.sum(axis=1) + explanation.base_values
    np.testing.assert_allclose(total, model.predict(X), atol=1e-3)


def test_the_useless_feature_gets_almost_no_shap_importance():
    X, y = toy_problem()
    model = XGBRegressor(n_estimators=50, max_depth=3).fit(X, y)
    importance = shap_importance(shap_values(model, X))
    assert importance.index[0] == "a"
    assert importance["b"] < 0.05 * importance["a"]


def test_importance_per_sensor_adds_up_value_mean_and_std():
    importance = pd.Series(
        {"s_1": 1.0, "s_1_mean": 2.0, "s_1_std": 0.5, "s_2": 0.1, "s_2_mean": 0.2, "s_2_std": 0.0}
    )
    per_sensor = importance_per_sensor(importance, ["s_1", "s_2"])
    assert per_sensor.to_dict() == pytest.approx({"s_1": 3.5, "s_2": 0.3})
    assert per_sensor.index[0] == "s_1"


def test_shuffling_an_unused_sensor_changes_nothing():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 30, 2)).astype(np.float32)
    y = 10 * X[:, -1, 0]  # only the last value of sensor 0 matters

    def predict_fn(windows):
        return 10 * windows[:, -1, 0]

    importance = permutation_importance(predict_fn, X, y, ["used", "unused"], n_repeats=3, seed=0)
    assert importance["unused"] == pytest.approx(0.0)
    assert importance["used"] > 5
    assert importance.index[0] == "used"


def test_permutation_does_not_modify_the_input():
    X = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    before = X.copy()
    permutation_importance(lambda w: w[:, -1, 0], X, X[:, -1, 0], ["a", "b"], n_repeats=2, seed=0)
    np.testing.assert_array_equal(X, before)
