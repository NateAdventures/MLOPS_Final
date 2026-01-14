%%writefile tests/test_features.py
import pandas as pd

from src.features.build_features import build_features


def test_build_features_adds_columns():
    df = pd.read_csv("data/raw/sample.csv")
    out = build_features(df)

    expected = ["AGE_BIN", "PAY_STATUS_MEAN", "BILL_TOTAL", "PAY_TOTAL"]
    for col in expected:
        assert col in out.columns


def test_build_features_no_nan_in_new_cols():
    df = pd.read_csv("data/raw/sample.csv")
    out = build_features(df)

    new_cols = ["PAY_STATUS_MEAN", "BILL_TOTAL", "PAY_TOTAL"]
    for col in new_cols:
        assert out[col].isna().sum() == 0
