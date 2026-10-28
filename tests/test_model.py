"""
tests/test_model.py
Tests for model training, evaluation, and artifact saving.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "feature_metadata.json"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"


@pytest.fixture(scope="module")
def df():
    if not PROCESSED_PATH.exists():
        pytest.skip("Processed dataset not found")
    df = pd.read_csv(PROCESSED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def pipeline():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found — run train_model.py first")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    if not METADATA_PATH.exists():
        pytest.skip("Metadata not found — run train_model.py first")
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def metrics():
    if not METRICS_PATH.exists():
        pytest.skip("Metrics not found — run train_model.py first")
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_model_artifact_exists():
    """The model joblib file must exist."""
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"


def test_metrics_artifact_exists():
    """The model metrics JSON must exist."""
    assert METRICS_PATH.exists(), f"Metrics file not found: {METRICS_PATH}"


def test_metadata_artifact_exists():
    """The feature metadata JSON must exist."""
    assert METADATA_PATH.exists(), f"Metadata file not found: {METADATA_PATH}"


def test_pipeline_loads(pipeline):
    """Pipeline must load without error."""
    assert pipeline is not None


def test_pipeline_has_predict(pipeline):
    """Pipeline must have a predict method."""
    assert hasattr(pipeline, "predict"), "Pipeline must have predict method"


def test_pipeline_has_predict_proba(pipeline):
    """Pipeline should support probability estimates."""
    assert hasattr(pipeline, "predict_proba"), \
        "Pipeline should have predict_proba method"


def test_metrics_have_required_keys(metrics):
    """Metrics JSON must have required top-level keys."""
    for key in ("best_model", "models", "train_rows", "test_rows"):
        assert key in metrics, f"Missing key in metrics: {key}"


def test_metrics_all_models_present(metrics):
    """Metrics must include all four model names."""
    expected_models = {
        "Logistic Regression", "Decision Tree",
        "Random Forest", "Gradient Boosting"
    }
    actual_models = set(metrics.get("models", {}).keys())
    assert expected_models.issubset(actual_models), (
        f"Missing models in metrics: {expected_models - actual_models}"
    )


def test_metrics_values_are_valid(metrics):
    """All metric values must be floats in [0, 1]."""
    for model_name, m in metrics.get("models", {}).items():
        for metric in ("accuracy", "precision", "recall", "f1"):
            val = m.get(metric)
            if val is not None:
                assert 0.0 <= val <= 1.0, (
                    f"{model_name} {metric}={val} out of [0,1]"
                )
        roc = m.get("roc_auc")
        if roc is not None:
            assert 0.0 <= roc <= 1.0, f"{model_name} roc_auc={roc} out of [0,1]"


def test_metadata_feature_columns(metadata):
    """Metadata must list feature columns."""
    feature_cols = metadata.get("feature_columns", [])
    assert len(feature_cols) > 0, "feature_columns must not be empty"


def test_metadata_has_teams(metadata):
    """Metadata must list all teams."""
    teams = metadata.get("all_teams", [])
    assert len(teams) > 5, "Expected at least 5 teams in metadata"


def test_pipeline_predict_shape(pipeline, df, metadata):
    """Pipeline must produce predictions with correct shape."""
    from feature_engineering import prepare_features, get_feature_columns
    df_feat = prepare_features(df.head(20))
    feature_cols = metadata.get("feature_columns", get_feature_columns())
    X = df_feat[feature_cols]
    preds = pipeline.predict(X)
    assert len(preds) == len(X), "Prediction count must match input rows"


def test_pipeline_probabilities_valid(pipeline, df, metadata):
    """Probabilities must be finite and in [0, 1]."""
    from feature_engineering import prepare_features, get_feature_columns
    df_feat = prepare_features(df.head(20))
    feature_cols = metadata.get("feature_columns", get_feature_columns())
    X = df_feat[feature_cols]
    proba = pipeline.predict_proba(X)
    assert np.all(np.isfinite(proba)), "Probabilities must be finite"
    assert np.all((proba >= 0) & (proba <= 1)), "Probabilities must be in [0, 1]"
    # Each row should sum to ~1
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), \
        f"Probability rows should sum to 1.0, got {row_sums[:5]}"


def test_historical_features_no_future_leakage(df):
    """
    Verify that historical features for match i only use matches 0..i-1.
    We check that a team's win_rate for its first-ever match is exactly the prior (0.5).
    """
    from feature_engineering import compute_historical_features

    # Take a small subset sorted by date
    df_small = df.sort_values("date").head(50).copy()
    df_feat = compute_historical_features(df_small)

    # The first match (row 0) must have win_rate = 0.5 (default prior)
    first_row = df_feat.iloc[0]
    assert first_row["team1_win_rate"] == 0.5, (
        f"First match team1_win_rate should be 0.5 (no prior history), "
        f"got {first_row['team1_win_rate']}"
    )
    assert first_row["team2_win_rate"] == 0.5, (
        f"First match team2_win_rate should be 0.5, got {first_row['team2_win_rate']}"
    )


def test_no_target_leakage_in_features(metadata):
    """Target column must not appear in feature_columns."""
    feature_cols = metadata.get("feature_columns", [])
    assert "target" not in feature_cols, \
        "'target' must NOT be in feature_columns (would be data leakage)"
    assert "winner" not in feature_cols, \
        "'winner' must NOT be in feature_columns (would be data leakage)"


def test_model_reloads_and_predicts(metadata, df):
    """Reloaded pipeline must produce predictions without error."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found")
    import joblib
    from feature_engineering import prepare_features, get_feature_columns

    pipeline = joblib.load(MODEL_PATH)
    df_feat = prepare_features(df.head(10))
    feature_cols = metadata.get("feature_columns", get_feature_columns())
    X = df_feat[feature_cols]
    preds = pipeline.predict(X)
    assert len(preds) == 10, "Should get 10 predictions"
    assert all(p in (0, 1) for p in preds), "Predictions must be 0 or 1"
