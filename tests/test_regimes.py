import numpy as np
import pandas as pd
import pytest

from turbofan.data import load_raw
from turbofan.regimes import fit_regime_scalers, regime_of, scale_by_regime


def two_regimes() -> pd.DataFrame:
    """Rows at two set points, with noisy settings; s_2 sits at 100 in one regime, 500 in the
    other."""
    rng = np.random.default_rng(0)
    n = 200
    high = np.arange(n) % 2 == 1  # every other row flies in the second regime
    return pd.DataFrame(
        {
            "setting_1": np.where(high, 42.0, 0.0) + rng.normal(0, 0.003, n),
            "setting_2": np.where(high, 0.84, 0.0) + rng.normal(0, 0.0003, n),
            "setting_3": np.where(high, 100.0, 60.0),
            "s_2": np.where(high, 500.0, 100.0) + rng.normal(0, 1, n),
        }
    )


def test_noisy_settings_are_rounded_to_their_set_point():
    regimes = regime_of(two_regimes())
    assert sorted(regimes.unique()) == ["0.0_0.0_60.0", "42.0_0.84_100.0"]


def test_each_regime_is_scaled_with_its_own_mean_and_spread():
    df = two_regimes()
    scaled = scale_by_regime(df, fit_regime_scalers(df, ["s_2"]), ["s_2"])
    by_regime = scaled.groupby(regime_of(df))["s_2"]
    # the jump from 100 to 500 is gone: each regime is centred on 0 with a spread of 1
    assert by_regime.mean().abs().max() < 1e-9
    assert by_regime.std(ddof=0).tolist() == pytest.approx([1.0, 1.0])
    assert df["s_2"].std() > 100  # the input is not modified


def test_a_regime_never_seen_in_training_is_refused():
    df = two_regimes()
    scalers = fit_regime_scalers(df[regime_of(df) == "0.0_0.0_60.0"], ["s_2"])
    with pytest.raises(ValueError, match="never seen in training"):
        scale_by_regime(df, scalers, ["s_2"])


@pytest.mark.needs_data
@pytest.mark.parametrize(
    ("subset", "expected"), [("FD001", 1), ("FD002", 6), ("FD003", 1), ("FD004", 6)]
)
def test_the_real_subsets_fly_at_one_or_six_regimes(subset, expected):
    train = load_raw(subset, "train")
    assert regime_of(train).nunique() == expected
    # the test engines fly at the same regimes as the training engines
    assert set(regime_of(load_raw(subset, "test"))) == set(regime_of(train))
