"""Data loading and cleaning for the Telco churn dataset.

"""
from __future__ import annotations

import pandas as pd


def load_raw_data(path: str, index_col: str) -> pd.DataFrame:
    """Load the raw churn CSV and set the customer ID as index."""
    df = pd.read_csv(path)
    df = df.set_index(index_col)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtype issues and impute missing values.

    TotalCharges arrives as a string in the raw export because a handful of
    brand-new customers (tenure == 0) have a blank value instead of "0.00".
    We coerce to numeric (blanks become NaN) and impute with the median.

    Runs identically at train time and inference time - this function has
    no dependency on the target column, so it's safe to reuse anywhere.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df