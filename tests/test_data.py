import zipfile

import pytest

from turbofan.config import COLUMNS, SENSOR_COLUMNS, load_params
from turbofan.data import extract_raw_files, load_raw, load_test_rul, sha256_of

# three rows in the raw format: 26 values separated by spaces, two spaces at the end of each line
TOY_ROWS = [
    "1 1 " + " ".join(["0.5"] * 24) + "  ",
    "1 2 " + " ".join(["0.6"] * 24) + "  ",
    "2 1 " + " ".join(["0.7"] * 24) + "  ",
]


def write_nested_archive(folder, inner_files):
    """Write an archive that contains CMAPSSData.zip, like the real download. Return both paths."""
    inner_path = folder / "inner_source.zip"
    with zipfile.ZipFile(inner_path, "w") as inner:
        for name, text in inner_files.items():
            inner.writestr(name, text)
    outer_path = folder / "outer.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.write(inner_path, "some folder/CMAPSSData.zip")
    return outer_path, inner_path


def test_sha256_of_known_content(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    # reference value of SHA-256("abc"), from the NIST test vectors
    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_extract_writes_text_files_when_fingerprint_matches(tmp_path):
    inner_files = {"train_FD001.txt": "1 1 0.5\n", "notes.pdf": "not a text file"}
    outer_path, inner_path = write_nested_archive(tmp_path, inner_files)
    raw_dir = tmp_path / "raw"

    extract_raw_files(outer_path, raw_dir, sha256_of(inner_path))

    assert (raw_dir / "train_FD001.txt").read_text() == "1 1 0.5\n"
    assert not (raw_dir / "notes.pdf").exists()


def test_extract_fails_when_fingerprint_differs(tmp_path):
    outer_path, _ = write_nested_archive(tmp_path, {"train_FD001.txt": "1 1 0.5\n"})
    raw_dir = tmp_path / "raw"

    with pytest.raises(ValueError, match="sha256"):
        extract_raw_files(outer_path, raw_dir, "0" * 64)

    assert not raw_dir.exists()


def test_load_raw_names_26_columns_despite_trailing_spaces(tmp_path):
    (tmp_path / "train_FD001.txt").write_text("\n".join(TOY_ROWS) + "\n")

    df = load_raw("FD001", "train", raw_dir=tmp_path)

    assert list(df.columns) == COLUMNS
    assert df.shape == (3, 26)
    assert df["unit_id"].tolist() == [1, 1, 2]
    assert df["cycle"].tolist() == [1, 2, 1]
    assert df["s_21"].tolist() == [0.5, 0.6, 0.7]
    assert df.isna().sum().sum() == 0


def test_load_raw_rejects_unknown_subset(tmp_path):
    with pytest.raises(ValueError, match="unknown subset"):
        load_raw("FD005", "train", raw_dir=tmp_path)


def test_load_test_rul_is_indexed_by_engine(tmp_path):
    (tmp_path / "RUL_FD001.txt").write_text("112 \n98 \n69 \n")

    rul = load_test_rul("FD001", raw_dir=tmp_path)

    assert rul.index.tolist() == [1, 2, 3]
    assert rul.tolist() == [112, 98, 69]


# reference values below were computed on the real files on 13/11/2026 (planning notes)


@pytest.mark.needs_data
@pytest.mark.parametrize(
    "subset, part, n_rows, n_engines, shortest, longest",
    [
        ("FD001", "train", 20631, 100, 128, 362),
        ("FD001", "test", 13096, 100, 31, 303),
        ("FD002", "train", 53759, 260, 128, 378),
        ("FD002", "test", 33991, 259, 21, 367),
        ("FD003", "train", 24720, 100, 145, 525),
        ("FD003", "test", 16596, 100, 38, 475),
        ("FD004", "train", 61249, 249, 128, 543),
        ("FD004", "test", 41214, 248, 19, 486),
    ],
)
def test_real_files_have_expected_sizes(subset, part, n_rows, n_engines, shortest, longest):
    df = load_raw(subset, part)
    lengths = df.groupby("unit_id")["cycle"].max()

    assert len(df) == n_rows
    assert df["unit_id"].nunique() == n_engines
    assert lengths.min() == shortest
    assert lengths.max() == longest


@pytest.mark.needs_data
def test_real_fd001_train_engines_and_first_row():
    df = load_raw("FD001", "train")
    lengths = df.groupby("unit_id")["cycle"].max()

    assert lengths.loc[1] == 192
    assert lengths.loc[2] == 287
    assert lengths.idxmin() == 39
    assert lengths.idxmax() == 69
    first_row = df.iloc[0]
    assert first_row["setting_1"] == pytest.approx(-0.0007)
    assert first_row["s_2"] == pytest.approx(641.82)
    assert first_row["s_21"] == pytest.approx(23.4190)


@pytest.mark.needs_data
def test_real_fd001_test_rul():
    rul = load_test_rul("FD001")

    assert len(rul) == 100
    assert rul.head(5).tolist() == [112, 98, 69, 82, 91]
    assert rul.min() == 7
    assert rul.max() == 145
    assert rul.mean() == pytest.approx(75.52)


@pytest.mark.needs_data
def test_selected_sensors_are_the_non_constant_ones_in_fd001():
    df = load_raw("FD001", "train")
    n_values = df.nunique()

    constant = [column for column in COLUMNS if n_values[column] == 1]
    assert constant == ["setting_3", "s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
    assert n_values["s_6"] == 2
    # s_6 only takes two values, so it is left out with the constant sensors
    varying_sensors = [s for s in SENSOR_COLUMNS if n_values[s] > 2]
    assert load_params()["prepare"]["sensors"] == varying_sensors
