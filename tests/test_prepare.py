import json

import pandas as pd
import pytest

from turbofan.config import PROCESSED_DIR, REPORTS_DIR, load_params
from turbofan.data import load_test_rul

PARTS = ["train", "val", "calib", "test"]


def read_processed() -> dict[str, pd.DataFrame]:
    """Return the four processed tables, or skip the test if prepare has not been run."""
    if not (PROCESSED_DIR / "test.parquet").exists():
        pytest.skip("processed data missing, run python -m pipeline.prepare")
    return {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in PARTS}


@pytest.mark.needs_data
def test_no_engine_is_in_two_training_parts():
    tables = read_processed()
    train, val, calib = (set(tables[name]["unit_id"]) for name in ["train", "val", "calib"])

    assert train & val == set()
    assert train & calib == set()
    assert val & calib == set()
    assert len(train | val | calib) == 100


@pytest.mark.needs_data
def test_processed_tables_match_the_report():
    tables = read_processed()
    report = json.loads((REPORTS_DIR / "prepare.json").read_text(encoding="utf-8"))

    for name in PARTS:
        assert len(tables[name]) == report["parts"][name]["n_rows"]
        assert sorted(tables[name]["unit_id"].unique()) == report["parts"][name]["units"]
    # the three training parts together are the whole training file of FD001
    assert sum(len(tables[name]) for name in ["train", "val", "calib"]) == 20631


@pytest.mark.needs_data
def test_train_sensors_are_centred_and_val_is_not_exactly():
    tables = read_processed()
    sensors = load_params()["prepare"]["sensors"]

    train_means = tables["train"][sensors].mean()
    val_means = tables["val"][sensors].mean()

    assert train_means.abs().max() < 1e-6
    assert tables["train"][sensors].std(ddof=0).tolist() == pytest.approx([1.0] * 14, abs=1e-6)
    # the scaler never saw the validation engines, so their means are not exactly 0
    assert val_means.abs().max() > 1e-3


@pytest.mark.needs_data
def test_test_targets_end_at_the_official_rul():
    test = read_processed()["test"]
    last_rows = test.loc[test.groupby("unit_id")["cycle"].idxmax()].set_index("unit_id")
    official = load_test_rul("FD001")

    assert last_rows["rul"].tolist() == official.tolist()
    assert last_rows["final_rul"].tolist() == official.tolist()
    # 11 test engines have more than 125 cycles left (planning notes, 13/11/2026)
    assert (last_rows["rul"] > 125).sum() == 11
    assert last_rows["rul_clipped"].max() == 125
