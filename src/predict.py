"""
src/predict.py
Cricket Match Winner Prediction — Prediction Module

Provides a reusable function to predict the winner of a T20I match.
Uses the saved pipeline (models/best_model.joblib) and metadata
(models/feature_metadata.json) to build features and return predictions.

Prediction timing: post-toss, pre-match.
No post-match information (winner, score, margins) is used.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"

DEFAULT_WIN_RATE = 0.5
RECENT_FORM_WINDOW = 5


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def load_pipeline(model_path: Optional[Path] = None):
    """Load the saved sklearn Pipeline."""
    if model_path is None:
        model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run 'python src/train_model.py' first."
        )
    pipeline = joblib.load(model_path)
    logger.info("Pipeline loaded from %s", model_path)
    return pipeline


def load_metadata(metadata_path: Optional[Path] = None) -> dict:
    """Load feature metadata."""
    if metadata_path is None:
        metadata_path = MODELS_DIR / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Run 'python src/train_model.py' first."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata


# ---------------------------------------------------------------------------
# Historical feature computation for a single prediction
# ---------------------------------------------------------------------------

def compute_prediction_features(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    prediction_date: pd.Timestamp,
    historical_df: pd.DataFrame,
) -> dict:
    """
    Compute the historical features for a single match prediction.

    Uses only matches from historical_df that occurred BEFORE prediction_date.
    No future information is used.

    Parameters
    ----------
    team1, team2 : str — participating teams
    venue : str
    toss_winner : str — must be team1 or team2
    toss_decision : str — 'bat' or 'field'
    prediction_date : pd.Timestamp
    historical_df : pd.DataFrame — the processed match dataset

    Returns
    -------
    dict of feature name → value (ready to build a one-row DataFrame)
    """
    # Encode toss winner relative to team1/team2
    if toss_winner == team1:
        toss_winner_enc = "team1"
    elif toss_winner == team2:
        toss_winner_enc = "team2"
    else:
        toss_winner_enc = "unknown"

    # Filter to matches strictly before prediction_date
    prior = historical_df[historical_df["date"] < prediction_date].copy()

    # -------------------------------------------------------
    # Team1 overall win rate from prior matches
    # -------------------------------------------------------
    t1_matches = prior[(prior["team1"] == team1) | (prior["team2"] == team1)]
    if len(t1_matches) > 0:
        t1_wins = sum(
            (1 if (r["team1"] == team1 and r["target"] == 1)
             or (r["team2"] == team1 and r["target"] == 0) else 0)
            for _, r in t1_matches.iterrows()
        )
        t1_win_rate = t1_wins / len(t1_matches)
    else:
        t1_win_rate = DEFAULT_WIN_RATE

    # -------------------------------------------------------
    # Team2 overall win rate
    # -------------------------------------------------------
    t2_matches = prior[(prior["team1"] == team2) | (prior["team2"] == team2)]
    if len(t2_matches) > 0:
        t2_wins = sum(
            (1 if (r["team1"] == team2 and r["target"] == 1)
             or (r["team2"] == team2 and r["target"] == 0) else 0)
            for _, r in t2_matches.iterrows()
        )
        t2_win_rate = t2_wins / len(t2_matches)
    else:
        t2_win_rate = DEFAULT_WIN_RATE

    # -------------------------------------------------------
    # Team1 recent form (last RECENT_FORM_WINDOW matches)
    # -------------------------------------------------------
    t1_recent_matches = t1_matches.sort_values("date").tail(RECENT_FORM_WINDOW)
    if len(t1_recent_matches) > 0:
        t1_recent_wins = sum(
            (1 if (r["team1"] == team1 and r["target"] == 1)
             or (r["team2"] == team1 and r["target"] == 0) else 0)
            for _, r in t1_recent_matches.iterrows()
        )
        t1_recent_form = t1_recent_wins / len(t1_recent_matches)
    else:
        t1_recent_form = DEFAULT_WIN_RATE

    # -------------------------------------------------------
    # Team2 recent form
    # -------------------------------------------------------
    t2_recent_matches = t2_matches.sort_values("date").tail(RECENT_FORM_WINDOW)
    if len(t2_recent_matches) > 0:
        t2_recent_wins = sum(
            (1 if (r["team1"] == team2 and r["target"] == 1)
             or (r["team2"] == team2 and r["target"] == 0) else 0)
            for _, r in t2_recent_matches.iterrows()
        )
        t2_recent_form = t2_recent_wins / len(t2_recent_matches)
    else:
        t2_recent_form = DEFAULT_WIN_RATE

    # -------------------------------------------------------
    # Head-to-head
    # -------------------------------------------------------
    h2h = prior[
        ((prior["team1"] == team1) & (prior["team2"] == team2))
        | ((prior["team1"] == team2) & (prior["team2"] == team1))
    ]
    if len(h2h) > 0:
        h2h_t1_wins = sum(
            (1 if (r["team1"] == team1 and r["target"] == 1)
             or (r["team2"] == team1 and r["target"] == 0) else 0)
            for _, r in h2h.iterrows()
        )
        h2h_t1_win_rate = h2h_t1_wins / len(h2h)
    else:
        h2h_t1_win_rate = DEFAULT_WIN_RATE

    return {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner_enc": toss_winner_enc,
        "toss_decision": toss_decision,
        "match_year": prediction_date.year,
        "team1_win_rate": round(t1_win_rate, 4),
        "team2_win_rate": round(t2_win_rate, 4),
        "team1_recent_form": round(t1_recent_form, 4),
        "team2_recent_form": round(t2_recent_form, 4),
        "h2h_team1_win_rate": round(h2h_t1_win_rate, 4),
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_inputs(
    team1: str,
    team2: str,
    toss_winner: str,
    toss_decision: str,
    metadata: dict,
) -> list[str]:
    """
    Validate user inputs against known categories.

    Returns a list of error messages (empty = valid).
    """
    errors = []

    if not team1 or not team1.strip():
        errors.append("Team 1 cannot be empty.")
    if not team2 or not team2.strip():
        errors.append("Team 2 cannot be empty.")
    if team1 and team2 and team1.strip() == team2.strip():
        errors.append("Team 1 and Team 2 must be different teams.")

    valid_toss = {team1, team2}
    if toss_winner not in valid_toss:
        errors.append(
            f"Toss winner must be one of the participating teams: "
            f"'{team1}' or '{team2}'. Got: '{toss_winner}'"
        )

    valid_decisions = metadata.get("toss_decision_options", ["bat", "field"])
    if toss_decision not in valid_decisions:
        errors.append(
            f"Toss decision must be one of {valid_decisions}. Got: '{toss_decision}'"
        )

    return errors


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_match(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    prediction_date: pd.Timestamp,
    pipeline=None,
    metadata: Optional[dict] = None,
    historical_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Predict the winner of a T20I match.

    Parameters
    ----------
    team1 : str — reference team (target=1 if team1 wins)
    team2 : str
    venue : str
    toss_winner : str — must be team1 or team2
    toss_decision : str — 'bat' or 'field'
    prediction_date : pd.Timestamp
    pipeline : fitted sklearn Pipeline (loaded if None)
    metadata : dict (loaded if None)
    historical_df : pd.DataFrame (loaded if None)

    Returns
    -------
    dict with keys:
        team1, team2, predicted_winner, team1_win_prob, team2_win_prob,
        model_name, errors, features_used
    """
    # Load artifacts if not provided
    if pipeline is None:
        pipeline = load_pipeline()
    if metadata is None:
        metadata = load_metadata()
    if historical_df is None:
        from data_loader import load_processed_data
        historical_df = load_processed_data()

    # Validate inputs
    errors = validate_inputs(team1, team2, toss_winner, toss_decision, metadata)
    if errors:
        return {
            "team1": team1,
            "team2": team2,
            "predicted_winner": None,
            "team1_win_prob": None,
            "team2_win_prob": None,
            "model_name": None,
            "errors": errors,
            "features_used": None,
        }

    # Compute features
    features = compute_prediction_features(
        team1, team2, venue, toss_winner, toss_decision,
        prediction_date, historical_df
    )

    feature_cols = metadata.get("feature_columns", [])
    X = pd.DataFrame([features])[feature_cols]

    # Predict
    prediction = int(pipeline.predict(X)[0])

    # Probability
    team1_prob = None
    team2_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)[0]
            # Class 0 = team2 wins, Class 1 = team1 wins
            classes = pipeline.classes_ if hasattr(pipeline, "classes_") else [0, 1]
            # Find class order from the final estimator
            final_clf = pipeline.named_steps.get("clf", pipeline)
            if hasattr(final_clf, "classes_"):
                classes = final_clf.classes_
            idx_0 = list(classes).index(0) if 0 in classes else 0
            idx_1 = list(classes).index(1) if 1 in classes else 1
            team1_prob = round(float(proba[idx_1]), 4)
            team2_prob = round(float(proba[idx_0]), 4)
        except Exception as exc:
            logger.warning("Probability extraction failed: %s", exc)

    predicted_winner = team1 if prediction == 1 else team2

    return {
        "team1": team1,
        "team2": team2,
        "predicted_winner": predicted_winner,
        "team1_win_prob": team1_prob,
        "team2_win_prob": team2_prob,
        "model_name": "Best Model (see model_metrics.json)",
        "errors": [],
        "features_used": features,
    }


if __name__ == "__main__":
    from data_loader import load_processed_data
    import datetime

    historical_df = load_processed_data()
    pipeline = load_pipeline()
    metadata = load_metadata()

    result = predict_match(
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
    print("\nPrediction Result:")
    for k, v in result.items():
        if k not in ("features_used",):
            print(f"  {k}: {v}")
