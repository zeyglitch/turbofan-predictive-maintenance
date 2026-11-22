import json

import pandas as pd
import pytest

from pipeline.prepare import prepare, split_by_subset
from turbofan.config import COLUMNS, ROOT, load_params
from turbofan.subsets import UNIT_OFFSET, load_subsets, load_subsets_test_rul


@pytest.fixture
def raw_dir(tmp_path, small_fleet):
    """The 8 synthetic engines saved as FD001 and as FD003, train and test, with RUL files."""
    for subset in ("FD001", "FD003"):
        for part in ("train", "test"):
            small_fleet[COLUMNS].to_csv(
                tmp_path / f"{part}_{subset}.txt", sep=" ", header=False, index=False
            )
        (tmp_path / f"RUL_{subset}.txt").write_text("\n".join(["10"] * 8), encoding="utf-8")
    return tmp_path


def test_engines_of_the_second_subset_get_new_numbers(raw_dir, small_fleet):
    fleet = load_subsets(["FD001", "FD003"], "train", raw_dir)
    assert len(fleet) == 2 * len(small_fleet)
    assert sorted(fleet["unit_id"].unique()) == list(range(1, 9)) + list(range(1001, 1009))
    # the first subset keeps its numbers: FD001 alone gives the same table as before
    alone = load_subsets(["FD001"], "train", raw_dir)
    assert alone["unit_id"].tolist() == small_fleet["unit_id"].tolist()


def test_test_rul_follows_the_same_numbers(raw_dir):
    rul = load_subsets_test_rul(["FD001", "FD003"], raw_dir)
    assert rul.index.tolist() == list(range(1, 9)) + list(range(1001, 1009))
    assert (rul == 10).all()


def test_each_subset_is_split_on_its_own(raw_dir):
    fleet = load_subsets(["FD001", "FD003"], "train", raw_dir)
    parts = split_by_subset(fleet, ["FD001", "FD003"], {"n_val": 2, "n_calib": 2, "seed": 42})
    for name, expected in [("train", 4), ("val", 2), ("calib", 2)]:
        units = parts[name]["unit_id"].unique()
        # the same number of engines from each subset, never one engine in two parts
        assert (units < UNIT_OFFSET).sum() == expected
        assert (units >= UNIT_OFFSET).sum() == expected
    all_units = pd.concat(parts.values())["unit_id"]
    assert all_units.nunique() == 16


@pytest.mark.needs_data
def test_fd001_plus_fd003_makes_200_engines_and_keeps_the_fd001_split(tmp_path):
    params = load_params()
    report = prepare(
        ["FD001", "FD003"], params, tmp_path, tmp_path / "scaler.joblib", tmp_path / "r.json"
    )
    parts = report["parts"]
    assert [parts[name]["n_engines"] for name in ("train", "val", "calib", "test")] == [
        140,
        30,
        30,
        200,
    ]
    # the FD001 engines land in the same parts as in the DVC pipeline (reports/prepare.json)
    pipeline = json.loads((ROOT / "reports" / "prepare.json").read_text(encoding="utf-8"))
    for name in ("train", "val", "calib"):
        fd001 = [unit for unit in parts[name]["units"] if unit < UNIT_OFFSET]
        assert fd001 == pipeline["parts"][name]["units"]
