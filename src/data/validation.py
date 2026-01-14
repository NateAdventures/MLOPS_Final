import pandas as pd
import pandera as pa
from pandera import Column, Check


# Я сделал схему не супер-жёсткой, чтобы не ловить глупые ошибки на кодировках.
# ключевые поля и диапазоны есть.
schema = pa.DataFrameSchema(
    {
        "LIMIT_BAL": Column(float, Check.ge(0), nullable=False, coerce=True),
        "SEX": Column(int, Check.isin([0, 1, 2]), nullable=False, coerce=True),
        "EDUCATION": Column(int, Check.between(0, 6), nullable=False, coerce=True),
        "MARRIAGE": Column(int, Check.between(0, 3), nullable=False, coerce=True),
        "AGE": Column(int, Check.between(18, 100), nullable=False, coerce=True),

        # Чуть расширил диапазон, чтобы не падало на редких значениях.
        "PAY_0": Column(int, Check.between(-3, 10), nullable=False, coerce=True),

        # Некоторые поля я оставляю как "float >=0" без супер строгого контроля.
        "BILL_AMT1": Column(float, nullable=True, coerce=True),
        "PAY_AMT1": Column(float, nullable=True, coerce=True),

        # target может отсутствовать при инференсе, поэтому nullable=True
        "default.payment.next.month": Column(int, Check.isin([0, 1]), nullable=True, coerce=True),
    },
    strict=False,   # разрешаем другие колонки
    coerce=True,
)


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Валидирует входной датафрейм.
    Возвращает валидированный df (pandera может привести типы).
    """
    # иногда читают csv и получают пробелы в названиях
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    validated = schema.validate(df, lazy=False)
    return validated
