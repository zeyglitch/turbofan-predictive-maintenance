"""Download the C-MAPSS archive and extract its text files into data/raw."""

import tempfile
from pathlib import Path

from turbofan.config import RAW_DIR, load_params
from turbofan.data import SUBSETS, download_archive, extract_raw_files, sha256_of


def main() -> None:
    params = load_params()["data"]

    expected_files = [
        RAW_DIR / f"{prefix}_{subset}.txt"
        for subset in SUBSETS
        for prefix in ("train", "test", "RUL")
    ]
    if all(path.exists() for path in expected_files):
        print(f"the {len(expected_files)} raw files are already in {RAW_DIR}, nothing to download")
        return

    # the archives are only needed during extraction, so they go to a temporary folder
    with tempfile.TemporaryDirectory() as temporary_dir:
        print(f"downloading {params['url']}")
        archive = download_archive(params["url"], Path(temporary_dir) / "cmapss.zip")
        print(f"downloaded {archive.stat().st_size} bytes, sha256 {sha256_of(archive)}")
        extract_raw_files(archive, RAW_DIR, params["inner_sha256"])

    print(f"wrote {len(list(RAW_DIR.glob('*.txt')))} text files to {RAW_DIR}")


if __name__ == "__main__":
    main()
