"""Paths, column names and parameter loading shared by the whole project."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# the raw files have no header: these are the 26 columns, in file order
INDEX_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    """Return the content of params.yaml as a nested dictionary."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)
