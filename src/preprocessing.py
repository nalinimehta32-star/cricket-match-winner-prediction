"""
src/preprocessing.py
Cricket Match Winner Prediction — Preprocessing Pipeline

Builds a reusable scikit-learn ColumnTransformer that:
- OneHotEncodes categorical features (with handle_unknown='ignore')
- StandardScales numeric features

The pipeline is fitted only on training data (never on test data).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import get_categorical_features, get_numeric_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """
    Build the feature preprocessor.

    Categorical features:
        OneHotEncoder with handle_unknown='ignore' so that unseen
        team names or venues at prediction time do not cause errors.

    Numeric features:
        StandardScaler to normalise win rates and form values to
        zero mean and unit variance (helps Logistic Regression).

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessor.
    """
    cat_features = get_categorical_features()
    num_features = get_numeric_features()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_features),
            ("num", numeric_transformer, num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically.

    Train set: earliest (1 - test_fraction) of matches.
    Test set:  latest test_fraction of matches.

    This prevents future match outcomes from influencing training,
    which would constitute data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset sorted by date.
    test_fraction : float
        Proportion of matches to hold out as test set (default 0.20).

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames
    """
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_fraction))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        "Chronological split: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        len(train_df),
        train_df["date"].min().date() if not train_df.empty else "N/A",
        train_df["date"].max().date() if not train_df.empty else "N/A",
        len(test_df),
        test_df["date"].min().date() if not test_df.empty else "N/A",
        test_df["date"].max().date() if not test_df.empty else "N/A",
    )

    return train_df, test_df


if __name__ == "__main__":
    from data_loader import load_processed_data
    from feature_engineering import get_feature_columns, prepare_features

    df = load_processed_data()
    df_feat = prepare_features(df)
    train_df, test_df = chronological_split(df_feat)

    features = get_feature_columns()
    X_train = train_df[features]
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)

    print(f"Training features shape: {X_train_transformed.shape}")
    print(f"Feature names (first 10): {preprocessor.get_feature_names_out()[:10]}")
