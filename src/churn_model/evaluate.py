"""
Returns a structured result instead of printing - printing is a UI concern
and belongs in train.py's logging or the Streamlit app, not buried in the
metric computation itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    roc_auc: float
    confusion: np.ndarray
    report: str
    y_pred: np.ndarray
    y_prob: np.ndarray


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return EvaluationResult(
        accuracy=accuracy_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_prob),
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred),
        y_pred=y_pred,
        y_prob=y_prob,
    )