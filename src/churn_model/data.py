"""Data loading and cleaning for the Telco churn dataset.

Maps to notebook cells 2-3 ("read csv" + "correcting mismatching datatypes").
"""
from __future__ import annotations

import pandas as pd


def load_raw_data(path: str, index_col: str) -> pd.DataFrame:
    """Load the raw churn CSV and set the customer ID as index."""
    df = pd.read_csv(path)
    df = df.set_index(index_col)
    return df


def clean_data(
    df: pd.DataFrame, total_charges_median: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Fix dtype issues and impute missing values.

    TotalCharges arrives as a string in the raw export because a handful of
    brand-new customers (tenure == 0) have a blank value instead of "0.00".
    We coerce to numeric (blanks become NaN) and impute with the median.

    total_charges_median:
        None (training): compute the median from this batch and return it,
        so it can be persisted alongside the model.
        A float (inference): reuse the training-time median instead of
        recomputing it. A single incoming request is one row - the median
        of one (possibly NaN) value is meaningless, so recomputing at
        inference time silently breaks. Always pass the stored training
        value here in the serving path.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if total_charges_median is None:
        total_charges_median = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)
    return df, total_charges_median