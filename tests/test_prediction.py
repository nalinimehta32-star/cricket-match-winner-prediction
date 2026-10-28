"""
tests/test_prediction.py
Tests for the prediction module and input validation.
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
METADATA_PATH = PROJECT_ROOT / "models" / "feature_metadata.json"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"


@pytest.fixture(scope="module")
def pipeline():
    if not MODEL_PATH.exists():
        pytest.skip("Model not found — run train_model.py")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    if not METADATA_PATH.exists():
        pytest.skip("Metadata not found — run train_model.py")
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def historical_df():
    if not PROCESSED_PATH.exists():
        pytest.skip("Processed dataset not found")
    df = pd.read_csv(PROCESSED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def sample_prediction(pipeline, metadata, historical_df):
    """Make a sample prediction for reuse in multiple tests."""
    from predict import predict_match
    return predict_match(
        team1="India",
        team2="Australia",
        venue="Melbourne Cricket Ground",
        toss_winner="India",
        toss_decision="bat",
        prediction_date=pd.Timestamp("2024-01-01"),
        pipeline=pipeline,
        metadata=metadata,
        historical_df=historical_df,
    )


def test_prediction_returns_dict(sample_prediction):
    """predict_match must return a dict."""
    assert isinstance(sample_prediction, dict)


def test_prediction_has_required_keys(sample_prediction):
    """Prediction dict must have all required keys."""
    required = [
        "team1", "team2", "predicted_winner",
        "team1_win_prob", "team2_win_prob", "errors"
    ]
    for key in required:
        assert key in sample_prediction, f"Missing key: {key}"


def test_prediction_no_errors(sample_prediction):
    """Valid inputs must produce no errors."""
    assert sample_prediction["errors"] == [], (
        f"Unexpected errors: {sample_prediction['errors']}"
    )


def test_predicted_winner_is_one_of_teams(sample_prediction):
    """Predicted winner must be one of the two input teams."""
    winner = sample_prediction["predicted_winner"]
    assert winner in ("India", "Australia"), (
        f"Predicted winner '{winner}' is not one of the input teams"
    )


def test_probabilities_sum_to_one(sample_prediction):
    """Team1 + Team2 win probabilities must sum to approximately 1.0."""
    t1 = sample_prediction.get("team1_win_prob")
    t2 = sample_prediction.get("team2_win_prob")
    if t1 is not None and t2 is not None:
        assert abs(t1 + t2 - 1.0) < 0.01, (
            f"Probabilities {t1} + {t2} = {t1+t2} should sum to ~1.0"
        )


def test_probabilities_in_valid_range(sample_prediction):
    """Probabilities must be in [0, 1]."""
    t1 = sample_prediction.get("team1_win_prob")
    t2 = sample_prediction.get("team2_win_prob")
    if t1 is not None:
        assert 0.0 <= t1 <= 1.0, f"team1_win_prob={t1} out of range"
    if t2 is not None:
        assert 0.0 <= t2 <= 1.0, f"team2_win_prob={t2} out of range"


def test_probabilities_are_finite(sample_prediction):
    """Probabilities must be finite (not NaN or Inf)."""
    t1 = sample_prediction.get("team1_win_prob")
    t2 = sample_prediction.get("team2_win_prob")
    if t1 is not None:
        assert np.isfinite(t1), f"team1_win_prob is not finite: {t1}"
    if t2 is not None:
        assert np.isfinite(t2), f"team2_win_prob is not finite: {t2}"


def test_team1_equals_team2_rejected(pipeline, metadata, historical_df):
    """Same team for team1 and team2 must be rejected."""
    from predict import predict_match
    result = predict_match(
        team1="India",
        team2="India",
        venue="Eden Gardens",
        toss_winner="India",
        toss_decision="bat",
        prediction_date=pd.Timestamp("2024-01-01"),
        pipeline=pipeline,
        metadata=metadata,
        historical_df=historical_df,
    )
    assert len(result["errors"]) > 0, "Should have rejected team1 == team2"


def test_invalid_toss_winner_rejected(pipeline, metadata, historical_df):
    """Toss winner that is not one of the playing teams must be rejected."""
    from predict import predict_match
    result = predict_match(
        team1="India",
        team2="Australia",
        venue="Eden Gardens",
        toss_winner="England",  # Not playing
        toss_decision="bat",
        prediction_date=pd.Timestamp("2024-01-01"),
        pipeline=pipeline,
        metadata=metadata,
        historical_df=historical_df,
    )
    assert len(result["errors"]) > 0, "Should reject toss_winner not in [team1, team2]"


def test_unknown_venue_handled_safely(pipeline, metadata, historical_df):
    """An unseen venue must not crash the predictor."""
    from predict import predict_match
    result = predict_match(
        team1="India",
        team2="Australia",
        venue="Totally Unknown Stadium XYZABC 99999",
        toss_winner="India",
        toss_decision="bat",
        prediction_date=pd.Timestamp("2024-01-01"),
        pipeline=pipeline,
        metadata=metadata,
        historical_df=historical_df,
    )
    # Should succeed (OneHotEncoder with handle_unknown='ignore')
    assert result["errors"] == [], (
        f"Unknown venue caused errors: {result['errors']}"
    )
    assert result["predicted_winner"] in ("India", "Australia")


def test_team_order_reversal_consistency(pipeline, metadata, historical_df):
    """
    Reversing team1/team2 must reverse the win probabilities appropriately.
    If P(India wins | India=team1) = p, then P(India wins | India=team2) ≈ 1-p
    (not exactly 1-p due to team-specific historical features, but Australia's
    probability should change accordingly).
    """
    from predict import predict_match

    result_a = predict_match(
        team1="India", team2="Australia",
        venue="Melbourne Cricket Ground",
        toss_winner="India", toss_decision="bat",
        prediction_date=pd.Timestamp("2024-01-01"),
        pipeline=pipeline, metadata=metadata, historical_df=historical_df,
    )
    result_b = predict_match(
        team1="Australia", team2="India",
        venue="Melbourne Cricket Ground",
        toss_winner="India", toss_decision="bat",
        prediction_date=pd.Timestamp("2024-01-01"),
        pipeline=pipeline, metadata=metadata, historical_df=historical_df,
    )

    t1a = result_a.get("team1_win_prob")  # P(India wins)
    t1b = result_b.get("team1_win_prob")  # P(Australia wins)

    if t1a is not None and t1b is not None:
        # They should sum to approximately 1
        assert abs(t1a + t1b - 1.0) < 0.05, (
            f"Reversed probabilities {t1a:.3f} + {t1b:.3f} should sum to ~1.0"
        )


def test_validate_inputs_directly(metadata):
    """Test the validate_inputs function directly."""
    from predict import validate_inputs

    # Valid inputs
    errors = validate_inputs("India", "Australia", "India", "bat", metadata)
    assert errors == [], f"Expected no errors, got: {errors}"

    # Same team
    errors = validate_inputs("India", "India", "India", "bat", metadata)
    assert len(errors) > 0

    # Toss winner not a team
    errors = validate_inputs("India", "Australia", "England", "bat", metadata)
    assert len(errors) > 0

    # Invalid toss decision
    errors = validate_inputs("India", "Australia", "India", "flip", metadata)
    assert len(errors) > 0


def test_streamlit_app_imports():
    """Streamlit app module must be importable without error."""
    import importlib.util
    app_path = PROJECT_ROOT / "app.py"
    if not app_path.exists():
        pytest.skip("app.py not found")

    spec = importlib.util.spec_from_file_location("app_module", app_path)
    # We can't run Streamlit in tests, but we can check the file parses OK
    with open(app_path, encoding="utf-8") as f:
        source = f.read()

    try:
        compile(source, "app.py", "exec")
    except SyntaxError as e:
        pytest.fail(f"app.py has a syntax error: {e}")
