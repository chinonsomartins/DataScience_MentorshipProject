"""End-to-end training pipeline.

Run directly (from the project root, with the venv active):
    python -m churn_model.train --config configs/config.yaml

This IS the notebook, restructured: load -> clean -> select features ->
split -> train (both models) -> evaluate -> persist artifacts. The
difference is every step's inputs/outputs are explicit function
arguments instead of implicit kernel state.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import yaml
from sklearn.model_selection import train_test_split

from churn_model.data import clean_data, load_raw_data
from churn_model.evaluate import evaluate_classifier
from churn_model.features import build_feature_matrix, select_features
from churn_model.models import train_logistic_regression, train_xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    target_col = cfg["data"]["target_col"]

    logger.info("Loading raw data from %s", cfg["data"]["raw_path"])
    df = load_raw_data(cfg["data"]["raw_path"], cfg["data"]["index_col"])
    df = clean_data(df)

    logger.info("Selecting features (chi-square + correlation) — training-time only")
    selected = select_features(
        df,
        target_col=target_col,
        top_n_categorical=cfg["feature_selection"]["top_n_categorical"],
        correlation_threshold=cfg["feature_selection"]["correlation_threshold"],
    )
    logger.info("Selected categorical: %s", selected.categorical)
    logger.info("Selected numerical: %s", selected.numerical)

    X, y = build_feature_matrix(df, selected, target_col=target_col, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )

    logger.info("Training logistic regression baseline")
    log_model = train_logistic_regression(X_train, y_train, cfg["models"]["logistic_regression"])
    log_result = evaluate_classifier(log_model, X_test, y_test)
    logger.info(
        "Logistic Regression — accuracy: %.4f | ROC-AUC: %.4f",
        log_result.accuracy,
        log_result.roc_auc,
    )

    logger.info("Training XGBoost")
    xgb_model = train_xgboost(X_train, y_train, cfg["models"]["xgboost"])
    xgb_result = evaluate_classifier(xgb_model, X_test, y_test)
    logger.info(
        "XGBoost — accuracy: %.4f | ROC-AUC: %.4f", xgb_result.accuracy, xgb_result.roc_auc
    )

    # --- This step did not exist in the notebook. Without it, there is
    # nothing on disk for an API to load. ---
    artifacts_dir = Path(cfg["artifacts"]["model_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_model, artifacts_dir / cfg["artifacts"]["xgboost_filename"])
    joblib.dump(log_model, artifacts_dir / cfg["artifacts"]["logistic_filename"])
    joblib.dump(selected, artifacts_dir / cfg["artifacts"]["feature_metadata_filename"])

    logger.info("Artifacts saved to %s", artifacts_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)