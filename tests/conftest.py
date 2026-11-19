import numpy as np
import pandas as pd
import pytest

from turbofan.config import COLUMNS, RAW_DIR


def pytest_collection_modifyitems(config, items):
    """Skip the tests marked needs_data when the real files have not been downloaded."""
    if (RAW_DIR / "train_FD001.txt").exists():
        return
    skip_needs_data = pytest.mark.skip(reason="real data missing, run python -m pipeline.download")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip_needs_data)


@pytest.fixture
def small_fleet() -> pd.DataFrame:
    """Eight synthetic engines in the raw format (26 columns), followed until failure.

    s_2 rises and s_11 falls as the engine wears out, s_1 is constant, the rest is noise.
    """
    rng = np.random.default_rng(0)
    engines = []
    for unit_id in range(1, 9):
        n_cycles = int(rng.integers(40, 81))
        cycle = np.arange(1, n_cycles + 1)
        wear = cycle / n_cycles  # goes from almost 0 to 1 at failure
        engine = pd.DataFrame({"unit_id": unit_id, "cycle": cycle})
        for column in COLUMNS[2:]:
            engine[column] = rng.normal(size=n_cycles)
        engine["s_1"] = 100.0
        engine["s_2"] = 5 * wear + rng.normal(scale=0.3, size=n_cycles)
        engine["s_11"] = -5 * wear + rng.normal(scale=0.3, size=n_cycles)
        engines.append(engine)
    return pd.concat(engines, ignore_index=True)
