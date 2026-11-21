
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_data import clean_data


def sample_data():
    return pd.DataFrame({
        "record_id": ["R001", "R001", "R002", "R003"],
        "date": ["2026-01-01"] * 4,
        "stops": [10, 10, -1, 20],
        "delay_logged_by_conductor": [5, 5, 8, 12],
        "delay_min": [5, 5, 8, 12],
    })


def test_duplicate_removal():
    result = clean_data(sample_data())
    assert len(result) == 3


def test_invalid_stops_fixed():
    result = clean_data(sample_data())
    assert (result["stops"] > 0).all()


def test_median_stop_imputation():
    result = clean_data(sample_data())
    assert result.loc[result["record_id"] == "R002", "stops"].iloc[0] == 15


def test_leakage_column_removed():
    result = clean_data(sample_data())
    assert "delay_logged_by_conductor" not in result.columns


def test_target_preserved():
    result = clean_data(sample_data())
    assert "delay_min" in result.columns
