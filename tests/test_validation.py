%%writefile tests/test_validation.py
import pandas as pd
import pytest

from src.data.validation import validate_df


def test_validation_ok():
    df = pd.read_csv("data/raw/sample.csv")
    # просто проверяем, что не падает
    validate_df(df)


def test_validation_fails_on_bad_age():
    df = pd.read_csv("data/raw/sample.csv")
    # делаем заведомо плохое значение
    df.loc[df.index[0], "AGE"] = 200

    with pytest.raises(Exception):
        validate_df(df)
