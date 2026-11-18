from turbofan.config import COLUMNS, load_params


def test_columns_are_26_unique_names():
    # 2 index columns + 3 operational settings + 21 sensors
    assert len(COLUMNS) == 26
    assert len(set(COLUMNS)) == 26


def test_load_params_reads_rul_cap():
    params = load_params()
    # value set in params.yaml, section prepare
    assert params["prepare"]["rul_cap"] == 125
