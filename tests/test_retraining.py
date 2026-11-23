import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from turbofan.config import load_params
from turbofan.networks import build_network
from turbofan.retraining import decide

SETTINGS = {"drift_share_threshold": 0.3, "min_improvement": 0.02, "max_degradation": 0.10}
CHAMPION = {"all": 30.0, "FD001": 15.0, "FD003": 40.0}


def test_a_better_challenger_is_promoted():
    challenger = {"all": 20.0, "FD001": 15.5, "FD003": 24.0}
    decision = decide(0.93, CHAMPION, challenger, SETTINGS)
    assert decision["retrain"] and decision["promote"]
    assert decision["improvement"] == pytest.approx(1 / 3)
    assert decision["fd001_change"] == pytest.approx(0.5 / 15)


def test_a_challenger_not_better_enough_is_not_promoted():
    challenger = {"all": 29.7, "FD001": 15.0, "FD003": 39.0}  # 1 % better, 2 % required
    decision = decide(0.93, CHAMPION, challenger, SETTINGS)
    assert decision["retrain"] and not decision["promote"]
    assert "less than the 2% required" in decision["reason"]


def test_a_challenger_worse_on_fd001_is_not_promoted_even_if_better_overall():
    challenger = {"all": 20.0, "FD001": 16.5, "FD003": 23.0}  # FD001 10 % worse: not allowed
    decision = decide(0.93, CHAMPION, challenger, SETTINGS)
    assert not decision["promote"]
    assert "FD001 worse by 10.0%" in decision["reason"]


def test_without_drift_nothing_is_trained():
    decision = decide(0.14, None, None, SETTINGS)  # no scores: nothing was trained
    assert decision == {
        "retrain": False,
        "promote": False,
        "improvement": None,
        "fd001_change": None,
        "reason": "drift share 0.14 below 0.3: no retraining",
    }


def test_the_rule_uses_the_thresholds_of_params():
    settings = load_params()["retraining"] | load_params()["monitoring"]
    assert settings["min_improvement"] == 0.02
    assert settings["max_degradation"] == 0.10
    assert settings["drift_share_threshold"] == 0.3


def test_validation_scores_are_split_by_subset():
    from flows.retrain_flow import score_on_validation_engines

    sensors = ["s_2", "s_3"]
    rng = np.random.default_rng(0)
    # two engines of the first subset, one of the second (renumbered from 1001)
    engines = pd.DataFrame(
        {
            "unit_id": [1] * 20 + [2] * 20 + [1001] * 20,
            "cycle": list(range(1, 21)) * 3,
            "s_2": rng.normal(size=60),
            "s_3": rng.normal(size=60),
            "rul_clipped": list(range(19, -1, -1)) * 3,
        }
    )
    bundle = {
        "model": build_network("lstm", 2, {"hidden_size": 4, "num_layers": 1, "dropout": 0.0}),
        "scaler": StandardScaler().fit(engines[sensors]),
        "metadata": {"sensors": sensors, "rul_cap": 125, "window_length": 5},
    }
    scores = score_on_validation_engines(bundle, engines, ["FD001", "FD003"])
    assert set(scores) == {"FD001", "FD003", "all"}
    # "all" weighs every cycle: it lies between the two subsets
    low, high = sorted([scores["FD001"], scores["FD003"]])
    assert low <= scores["all"] <= high
