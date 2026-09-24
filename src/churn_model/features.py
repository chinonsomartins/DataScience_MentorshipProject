"""Feature selection and encoding.

 (chi-square test, correlation test, feature
selection, one-hot encoding).

Two functions, two different lifecycles - this split is the whole point:

- select_features(): needs the TARGET column. Only ever runs during
  training. Never call this from the serving/API path - you won't have
  a label for an incoming request.
- build_feature_matrix(): safe to call at both train time (fit=True)
  and inference time (fit=False). At inference time it reindexes to the
  exact column set seen during training, so a single-row request that's
  missing some categories still produces a matrix the model recognizes.
  This replaces the notebook's bare `pd.get_dummies(X, drop_first=True)`,
  which silently breaks on any input whose categories don't match the
  full training set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


@dataclass
class SelectedFeatures:
    """Everything the inference step needs to know about training-time
    feature decisions. Serialized alongside the model."""

    categorical: list[str]
    numerical: list[str]
    encoded_columns: list[str] = field(default_factory=list)
    total_charges_median: float = 0.0


def select_features(
    df: pd.DataFrame,
    target_col: str,
    top_n_categorical: int = 7,
    correlation_threshold: float = 0.1,
) -> SelectedFeatures:
    """Rank categorical features by chi-square and numerical features by
    correlation with the target, returning the selected subset.

    TRAINING-TIME ONLY - requires the target column.
    """
    df_chi = df.copy()
    df_chi["_target_binary"] = df_chi[target_col].map({"Yes": 1, "No": 0})

    cat_cols = df_chi.select_dtypes(include="object").columns.drop(target_col)

    le = LabelEncoder()
    for col in cat_cols:
        df_chi[col] = le.fit_transform(df_chi[col])

    chi_scores, p_values = chi2(df_chi[cat_cols], df_chi["_target_binary"])
    chi_df = pd.DataFrame(
        {"Feature": cat_cols, "Chi_Square_Score": chi_scores, "p_value": p_values}
    ).sort_values(by="Chi_Square_Score", ascending=False)

    top_categorical = chi_df.head(top_n_categorical)["Feature"].tolist()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr_df = df[num_cols].copy()
    corr_df["_target_binary"] = df[target_col].map({"Yes": 1, "No": 0})
    corr_with_target = (
        corr_df.corr()["_target_binary"].abs().sort_values(ascending=False)
    )
    significant_numerical = corr_with_target[
        corr_with_target > correlation_threshold
    ].index.tolist()
    significant_numerical = [c for c in significant_numerical if c != "_target_binary"]

    return SelectedFeatures(categorical=top_categorical, numerical=significant_numerical)


def build_feature_matrix(
    df: pd.DataFrame,
    selected: SelectedFeatures,
    target_col: str,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """One-hot encode the selected features.

    fit=True  (training):  learns the resulting column set and stores it
                            on `selected.encoded_columns`.
    fit=False (inference):  reindexes to that stored column set, filling
                            any missing category-columns with 0. This is
                            what makes a single live request produce a
                            matrix shaped identically to training data.
    """
    feature_cols = selected.categorical + selected.numerical
    X = df[feature_cols]
    X = pd.get_dummies(X, drop_first=True)

    if fit:
        selected.encoded_columns = X.columns.tolist()
    else:
        X = X.reindex(columns=selected.encoded_columns, fill_value=0)

    y = None
    if target_col in df.columns:
        y = df[target_col].map({"Yes": 1, "No": 0})

    return X, y