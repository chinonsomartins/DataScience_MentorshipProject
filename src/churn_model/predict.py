"""Inference-time prediction logic.

This module contains ONLY what the API needs at request time: load
persisted artifacts once, then transform + predict per request. It
deliberately does not import select_features - that function needs a
target label and has no place in a serving path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from churn_model.data import clean_data
from churn_model.features import SelectedFeatures, build_feature_matrix


@dataclass
class ChurnArtifacts:
    model: object
    selected_features: SelectedFeatures


def load_artifacts(model_dir: str, model_filename: str, feature_metadata_filename: str) -> ChurnArtifacts:
    """Load the trained model and its feature metadata once, at process
    startup - not per-request. This is called once when the FastAPI app
    starts (see api/main.py's lifespan handler)."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / model_filename)
    selected_features = joblib.load(model_dir / feature_metadata_filename)
    return ChurnArtifacts(model=model, selected_features=selected_features)


def predict_one(record: dict, artifacts: ChurnArtifacts) -> dict:
    """Run one customer record through the full inference path.

    `record` is a single raw customer's fields, in the same shape as a row
    of the original CSV (minus customerID and Churn). Returns the churn
    probability and a thresholded label.
    """
    df = pd.DataFrame([record])

    df, _ = clean_data(df, total_charges_median=artifacts.selected_features.total_charges_median)

    X, _ = build_feature_matrix(
        df, artifacts.selected_features, target_col="Churn", fit=False
    )

    probability = float(artifacts.model.predict_proba(X)[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "churn_predicted": probability >= 0.5,
    }