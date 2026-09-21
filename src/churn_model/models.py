"""Model training. Maps to notebook cells 13 and 19.

Deliberately thin: these are wrappers, not classes, because there's no
extra behavior to encapsulate yet. Config drives hyperparameters (**params)
instead of hardcoding them here, so a hyperparameter sweep later just means
looping over config dicts, not editing this file.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict
) -> LogisticRegression:
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict
) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model