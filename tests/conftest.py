import pytest

from turbofan.config import RAW_DIR


def pytest_collection_modifyitems(config, items):
    """Skip the tests marked needs_data when the real files have not been downloaded."""
    if (RAW_DIR / "train_FD001.txt").exists():
        return
    skip_needs_data = pytest.mark.skip(reason="real data missing, run python -m pipeline.download")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip_needs_data)
