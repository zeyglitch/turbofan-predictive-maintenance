"""Download, check and load the raw C-MAPSS files."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from turbofan.config import COLUMNS, RAW_DIR

INNER_ARCHIVE_NAME = "CMAPSSData.zip"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def sha256_of(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file, as 64 hexadecimal characters."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_archive(url: str, destination: Path) -> Path:
    """Download the file at url to destination and return destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def extract_raw_files(archive: Path, raw_dir: Path, expected_inner_sha256: str) -> None:
    """Extract the text files into raw_dir, after checking the fingerprint of the inner archive."""
    # the downloaded archive contains a second archive, which contains the text files
    with zipfile.ZipFile(archive) as outer:
        inner_names = [name for name in outer.namelist() if name.endswith(INNER_ARCHIVE_NAME)]
        if len(inner_names) != 1:
            raise ValueError(
                f"expected one {INNER_ARCHIVE_NAME} in {archive.name}, found {inner_names}"
            )
        inner_path = archive.parent / INNER_ARCHIVE_NAME
        inner_path.write_bytes(outer.read(inner_names[0]))

    found_sha256 = sha256_of(inner_path)
    if found_sha256 != expected_inner_sha256:
        raise ValueError(
            f"{INNER_ARCHIVE_NAME} has sha256 {found_sha256}, expected {expected_inner_sha256}: "
            "the download is corrupted or the dataset has changed"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(inner_path) as inner:
        for name in inner.namelist():
            if name.endswith(".txt"):
                # keep only the file name, whatever folder it has inside the archive
                (raw_dir / Path(name).name).write_bytes(inner.read(name))


def load_raw(subset: str, part: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return the train or test file of a subset, one row per engine and cycle, 26 named columns."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset}, expected FD001 to FD004")
    if part not in ("train", "test"):
        raise ValueError(f"unknown part {part}, expected train or test")
    # values are separated by spaces and every line ends with spaces: \s+ handles both
    return pd.read_csv(raw_dir / f"{part}_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    """Return the true remaining useful life of each test engine, indexed by unit_id."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset}, expected FD001 to FD004")
    rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None, names=["rul"])["rul"]
    # the file has one value per line, in engine order: line 1 is engine 1
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit_id")
    return rul
