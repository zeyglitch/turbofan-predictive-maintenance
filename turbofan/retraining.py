"""The written rule that decides whether a challenger replaces the champion.

Scores are test RMSE on the last cycle of each engine: "all" for every test engine of the
subsets (200 for FD001 + FD003), plus one key per subset ("FD001", "FD003").
"""

import json

from turbofan.config import MODELS_DIR, REPORTS_DIR


def decide(
    drift_share: float, champion: dict | None, challenger: dict | None, settings: dict
) -> dict:
    """Return {"retrain", "promote", "improvement", "fd001_change", "reason"}.

    No retraining below the drift threshold. Then the challenger is promoted only if it lowers
    the RMSE on all test engines by at least min_improvement, and raises the RMSE on FD001 by
    less than max_degradation: a better average must not hide a worse model on the old engines.
    """
    if drift_share < settings["drift_share_threshold"]:
        return {
            "retrain": False,
            "promote": False,
            "improvement": None,
            "fd001_change": None,
            "reason": (
                f"drift share {drift_share:.2f} below {settings['drift_share_threshold']}: "
                "no retraining"
            ),
        }
    improvement = (champion["all"] - challenger["all"]) / champion["all"]
    fd001_change = (challenger["FD001"] - champion["FD001"]) / champion["FD001"]
    if improvement < settings["min_improvement"]:
        promote = False
        reason = (
            f"RMSE on all test engines improved by {improvement:.1%}, "
            f"less than the {settings['min_improvement']:.0%} required"
        )
    elif fd001_change >= settings["max_degradation"]:
        promote = False
        reason = (
            f"RMSE on FD001 worse by {fd001_change:.1%}, "
            f"not less than the {settings['max_degradation']:.0%} allowed"
        )
    else:
        promote = True
        reason = (
            f"RMSE on all test engines improved by {improvement:.1%} "
            f"and FD001 changed by {fd001_change:+.1%}"
        )
    return {
        "retrain": True,
        "promote": promote,
        "improvement": improvement,
        "fd001_change": fd001_change,
        "reason": reason,
    }


def champion_replaced() -> bool:
    """True when models/champion is no longer the champion of the DVC pipeline.

    The retraining flow promotes a challenger by replacing that folder, without DVC: the bundle
    then carries another registry version than reports/champion.json, written by `register`.
    """
    bundle = MODELS_DIR / "champion" / "metadata.json"
    pipeline = REPORTS_DIR / "champion.json"
    if not (bundle.exists() and pipeline.exists()):
        return False
    served = json.loads(bundle.read_text(encoding="utf-8"))["registry_version"]
    return served != json.loads(pipeline.read_text(encoding="utf-8"))["registry_version"]
