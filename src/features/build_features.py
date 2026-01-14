%%writefile src/features/build_features.py
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # AGE_BIN: простое биннирование
    if "AGE" in df.columns:
        bins = [0, 25, 35, 45, 55, 65, 200]
        labels = [0, 1, 2, 3, 4, 5]
        df["AGE_BIN"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=True).astype("float")

    # PAY_STATUS_MEAN
    pay_cols = [c for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in df.columns]
    if len(pay_cols) > 0:
        df["PAY_STATUS_MEAN"] = df[pay_cols].mean(axis=1)
    else:
        df["PAY_STATUS_MEAN"] = 0.0

    # BILLTOTAL
    bill_cols = [c for c in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"] if c in df.columns]
    if len(bill_cols) > 0:
        df["BILL_TOTAL"] = df[bill_cols].sum(axis=1)
    else:
        df["BILL_TOTAL"] = 0.0

    # PAYTOTAL
    pay_amt_cols = [c for c in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"] if c in df.columns]
    if len(pay_amt_cols) > 0:
        df["PAY_TOTAL"] = df[pay_amt_cols].sum(axis=1)
    else:
        df["PAY_TOTAL"] = 0.0

    # маленькая чистка NaN в новых колонках (просто на всякий)
    for col in ["AGE_BIN", "PAY_STATUS_MEAN", "BILL_TOTAL", "PAY_TOTAL"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
