"""
src/feature_engineering.py
Cricket Match Winner Prediction — Feature Engineering

Computes historical features using ONLY matches that occurred BEFORE
the match being predicted. No future information is used.

Features computed per match (at prediction time = post-toss, pre-match):
  1. team1                     (categorical)
  2. team2                     (categorical)
  3. venue                     (categorical)
  4. toss_winner               (categorical: "team1" or "team2")
  5. toss_decision             (categorical: "bat" or "field")
  6. match_year                (numeric)
  7. team1_win_rate            (numeric: historical win rate of team1)
  8. team2_win_rate            (numeric: historical win rate of team2)
  9. team1_recent_form         (numeric: last-5-match win rate of team1)
  10. team2_recent_form        (numeric: last-5-match win rate of team2)
  11. h2h_team1_win_rate       (numeric: head-to-head win rate for team1)

All historical statistics are computed via a chronological pass through
the dataset so that each match only sees outcomes of earlier matches.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"

# Fallback win rate for teams with no prior history
# (using 0.5 = equal chance, documented below)
DEFAULT_WIN_RATE = 0.5
RECENT_FORM_WINDOW = 5  # last N completed matches


def encode_toss_winner(row: pd.Series) -> str:
    """Encode toss_winner as 'team1' or 'team2' relative to team1/team2 in this row."""
    if pd.isna(row["toss_winner"]):
        return "unknown"
    if row["toss_winner"] == row["team1"]:
        return "team1"
    if row["toss_winner"] == row["team2"]:
        return "team2"
    return "unknown"


def compute_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical win rates and recent form for every match.

    The dataset is processed in chronological order. For match i:
    - Only matches 0..i-1 are used to compute team1/team2 statistics.
    - The current match outcome is excluded.
    - This guarantees no data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Processed match data, sorted by date (ascending).
        Must have columns: match_id, date, team1, team2, winner, target.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added historical feature columns.
    """
    df = df.copy()
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # Dictionaries to accumulate per-team and per-pair history
    # team_history[team] = list of (date, won: bool)
    team_history: dict[str, list[tuple]] = {}
    # pair_history[(teamA, teamB)] = list of (date, teamA_won: bool)
    pair_history: dict[tuple, list[tuple]] = {}

    feature_rows = []

    for idx, row in df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        match_date = row["date"]

        # -------------------------------------------------------
        # Compute team1 overall historical win rate
        # -------------------------------------------------------
        t1_hist = team_history.get(t1, [])
        if t1_hist:
            t1_win_rate = sum(w for _, w in t1_hist) / len(t1_hist)
        else:
            t1_win_rate = DEFAULT_WIN_RATE  # documented fallback

        # -------------------------------------------------------
        # Compute team2 overall historical win rate
        # -------------------------------------------------------
        t2_hist = team_history.get(t2, [])
        if t2_hist:
            t2_win_rate = sum(w for _, w in t2_hist) / len(t2_hist)
        else:
            t2_win_rate = DEFAULT_WIN_RATE

        # -------------------------------------------------------
        # Compute team1 recent form (last RECENT_FORM_WINDOW matches)
        # -------------------------------------------------------
        recent_t1 = t1_hist[-RECENT_FORM_WINDOW:]
        if recent_t1:
            t1_recent = sum(w for _, w in recent_t1) / len(recent_t1)
        else:
            t1_recent = DEFAULT_WIN_RATE

        # -------------------------------------------------------
        # Compute team2 recent form
        # -------------------------------------------------------
        recent_t2 = t2_hist[-RECENT_FORM_WINDOW:]
        if recent_t2:
            t2_recent = sum(w for _, w in recent_t2) / len(recent_t2)
        else:
            t2_recent = DEFAULT_WIN_RATE

        # -------------------------------------------------------
        # Head-to-head: team1 vs team2
        # We look up both orderings (t1,t2) and (t2,t1)
        # -------------------------------------------------------
        key = (min(t1, t2), max(t1, t2))
        h2h_hist = pair_history.get(key, [])
        if h2h_hist:
            # Each entry: (date, t1_won) where t1 = min(t1,t2) key ordering
            # We need to adjust for which team is team1 in this match
            t1_is_min = (t1 == min(t1, t2))
            h2h_t1_wins = sum(
                w if t1_is_min else (1 - w) for _, w in h2h_hist
            )
            h2h_t1_win_rate = h2h_t1_wins / len(h2h_hist)
        else:
            h2h_t1_win_rate = DEFAULT_WIN_RATE

        feature_rows.append({
            "team1_win_rate": round(t1_win_rate, 4),
            "team2_win_rate": round(t2_win_rate, 4),
            "team1_recent_form": round(t1_recent, 4),
            "team2_recent_form": round(t2_recent, 4),
            "h2h_team1_win_rate": round(h2h_t1_win_rate, 4),
        })

        # -------------------------------------------------------
        # Update histories AFTER computing features for this match
        # (guarantees this match's outcome is not used for itself)
        # -------------------------------------------------------
        t1_won = bool(row["target"] == 1)
        t2_won = not t1_won

        # Update team1 history
        if t1 not in team_history:
            team_history[t1] = []
        team_history[t1].append((match_date, int(t1_won)))

        # Update team2 history
        if t2 not in team_history:
            team_history[t2] = []
        team_history[t2].append((match_date, int(t2_won)))

        # Update head-to-head history
        t1_is_min = (t1 == min(t1, t2))
        if key not in pair_history:
            pair_history[key] = []
        pair_history[key].append((match_date, int(t1_is_min == t1_won)))

    features_df = pd.DataFrame(feature_rows, index=df.index)
    df = pd.concat([df, features_df], axis=1)

    logger.info("Historical features computed for %d matches", len(df))
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all features for model training / prediction.

    Steps:
    1. Compute historical features (chronological pass).
    2. Encode toss_winner as 'team1' or 'team2'.
    3. Add match_year.
    4. Return DataFrame with feature columns + target.
    """
    df = df.copy()

    # Ensure sorted by date
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # Historical statistics
    df = compute_historical_features(df)

    # Toss winner encoded relative to team1/team2
    df["toss_winner_enc"] = df.apply(encode_toss_winner, axis=1)

    # Match year
    df["match_year"] = df["date"].dt.year

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used for modelling."""
    return [
        "team1",
        "team2",
        "venue",
        "toss_winner_enc",
        "toss_decision",
        "match_year",
        "team1_win_rate",
        "team2_win_rate",
        "team1_recent_form",
        "team2_recent_form",
        "h2h_team1_win_rate",
    ]


def get_categorical_features() -> list[str]:
    """Categorical feature names."""
    return ["team1", "team2", "venue", "toss_winner_enc", "toss_decision"]


def get_numeric_features() -> list[str]:
    """Numeric feature names."""
    return [
        "match_year",
        "team1_win_rate",
        "team2_win_rate",
        "team1_recent_form",
        "team2_recent_form",
        "h2h_team1_win_rate",
    ]


if __name__ == "__main__":
    from data_loader import load_processed_data

    df = load_processed_data()
    df_feat = prepare_features(df)
    print(df_feat[get_feature_columns() + ["target"]].head(10))
    print(f"\nShape after feature engineering: {df_feat.shape}")
