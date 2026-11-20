import numpy as np
import pytest

from turbofan.conformal import conformal_quantile, empirical_coverage, prediction_interval


def errors_as_predictions(errors):
    """Truth 0 and prediction = error, so that |pred - true| is exactly the given error."""
    return np.zeros(len(errors)), np.asarray(errors, dtype=float)


def test_quantile_of_19_errors_is_the_18th():
    # k = ceil(20 * 0.9) = 18
    y_true, y_pred = errors_as_predictions(range(1, 20))
    assert conformal_quantile(y_true, y_pred, 0.90) == 18


def test_quantile_of_10_errors_is_the_largest():
    # k = ceil(11 * 0.9) = ceil(9.9) = 10
    y_true, y_pred = errors_as_predictions(range(1, 11))
    assert conformal_quantile(y_true, y_pred, 0.90) == 10


def test_quantile_when_k_is_larger_than_n_takes_the_largest_error():
    # k = ceil(6 * 0.9) = 6 > 5
    y_true, y_pred = errors_as_predictions(range(1, 6))
    assert conformal_quantile(y_true, y_pred, 0.90) == 5


def test_quantile_uses_absolute_errors_in_any_order():
    # errors 3, 1, 2 in both directions; k = ceil(4 * 0.5) = 2, the second smallest is 2
    assert conformal_quantile([10, 10, 10], [13, 9, 12], 0.50) == 2


def test_interval_is_bounded_to_0_and_the_cap():
    low, high = prediction_interval([5, 60, 120], q=10, rul_cap=125)
    np.testing.assert_array_equal(low, [0, 50, 110])
    np.testing.assert_array_equal(high, [15, 70, 125])


def test_coverage_counts_true_values_inside_their_interval():
    coverage = empirical_coverage([10, 50, 90], [0, 55, 80], [20, 65, 100])
    assert coverage == pytest.approx(2 / 3)


def test_bounds_of_the_interval_count_as_inside():
    assert empirical_coverage([10, 20], [10, 15], [15, 20]) == 1.0
