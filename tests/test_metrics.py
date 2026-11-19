import numpy as np
import pytest

from turbofan.config import load_params
from turbofan.data import load_test_rul
from turbofan.labels import clip_rul
from turbofan.metrics import alert_metrics, mae, nasa_score, regression_report, rmse

# three engines, errors d = pred - true = [-13, +10, 0]
Y_TRUE = [100, 50, 20]
Y_PRED = [87, 60, 20]


def test_rmse_by_hand():
    # sqrt((169 + 100 + 0) / 3)
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(np.sqrt(269 / 3))
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(9.4692, abs=1e-4)


def test_mae_by_hand():
    # (13 + 10 + 0) / 3
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(23 / 3)


def test_nasa_score_by_hand():
    # d = -13 gives exp(1) - 1, d = +10 gives exp(1) - 1, d = 0 gives 0
    assert nasa_score(Y_TRUE, Y_PRED) == pytest.approx(2 * (np.e - 1))
    assert nasa_score(Y_TRUE, Y_PRED) == pytest.approx(3.436564, abs=1e-6)


def test_nasa_score_punishes_late_predictions_more():
    too_optimistic = nasa_score([50], [70])  # d = +20
    too_cautious = nasa_score([50], [30])  # d = -20

    assert too_optimistic == pytest.approx(np.exp(2) - 1)
    assert too_cautious == pytest.approx(np.exp(20 / 13) - 1)
    assert too_optimistic > too_cautious


def test_perfect_prediction_scores_zero():
    assert regression_report([10, 20, 30], [10, 20, 30]) == {
        "rmse": 0.0,
        "mae": 0.0,
        "nasa_score": 0.0,
    }


def test_alert_metrics_by_hand():
    # threshold 30: true alerts are 10 and 25, predicted alerts are 20 and 28
    report = alert_metrics([10, 25, 40, 80], [20, 35, 28, 90], threshold=30)

    assert report == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": 0.5, "recall": 0.5}


def test_alert_metrics_without_any_predicted_alert():
    report = alert_metrics([10, 80], [90, 90], threshold=30)

    assert report["precision"] == 0.0
    assert report["recall"] == 0.0
    assert report["fn"] == 1


# reference values computed on the real file on 13/11/2026 (planning notes)


@pytest.mark.needs_data
def test_constant_prediction_on_fd001_test_gives_rmse_40():
    truth = clip_rul(load_test_rul("FD001"), cap=load_params()["prepare"]["rul_cap"])

    assert rmse(truth, np.full(100, 74.45)) == pytest.approx(40.073, abs=1e-3)
    assert (load_test_rul("FD001") <= 30).sum() == 25
    assert (load_test_rul("FD001") > 125).sum() == 11
