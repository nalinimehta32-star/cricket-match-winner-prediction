"""
tests/test_data.py
Tests for data loading and schema validation.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    load_processed_data,
    data_quality_report,
    check_duplicates,
    check_target_classes,
    check_toss_validity,
    check_winner_validity,
    validate_schema,
)

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"


@pytest.fixture(scope="module")
def df():
    """Load the processed dataset once for all tests in this module."""
    if not PROCESSED_PATH.exists():
        pytest.skip("Processed dataset not found — run acquire_data.py first")
    return load_processed_data(PROCESSED_PATH)


def test_dataset_loads(df):
    """Dataset must load without error and have at least 100 rows."""
    assert df is not None, "Dataset returned None"
    assert len(df) >= 100, f"Expected at least 100 rows, got {len(df)}"


def test_schema_columns(df):
    """Dataset must have all required columns."""
    required = [
        "match_id", "date", "season", "team1", "team2",
        "venue", "toss_winner", "toss_decision", "winner", "target",
    ]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"


def test_target_valid_classes(df):
    """Target column must contain only 0 and 1."""
    result = check_target_classes(df)
    invalid = result.get("invalid_values", [])
    assert len(invalid) == 0, f"Invalid target values found: {invalid}"
    counts = result.get("class_counts", {})
    assert 0 in counts and 1 in counts, "Target must have both class 0 and class 1"


def test_no_duplicate_match_ids(df):
    """There must be no duplicate match_id values."""
    result = check_duplicates(df)
    assert result["duplicate_ids"] == 0, (
        f"Found {result['duplicate_ids']} duplicate match IDs"
    )


def test_date_column_parsed(df):
    """Date column must be parseable as datetime."""
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), \
        "date column must be datetime type"
    bad_dates = df["date"].isna().sum()
    assert bad_dates == 0, f"Found {bad_dates} unparseable dates"


def test_toss_winner_is_one_of_teams(df):
    """toss_winner must always be one of team1 or team2."""
    result = check_toss_validity(df)
    invalid_count = result.get("invalid_toss_winner_count", -1)
    assert invalid_count == 0, (
        f"Found {invalid_count} rows where toss_winner is not team1 or team2"
    )


def test_winner_is_one_of_teams(df):
    """winner must always be one of team1 or team2."""
    result = check_winner_validity(df)
    invalid_count = result.get("invalid_winner_count", -1)
    assert invalid_count == 0, (
        f"Found {invalid_count} rows where winner is not team1 or team2"
    )


def test_target_matches_winner(df):
    """target=1 must correspond to team1 winning, target=0 to team2."""
    mask_team1_won = df["winner"] == df["team1"]
    mask_target_1 = df["target"] == 1
    mismatches = (mask_team1_won != mask_target_1).sum()
    assert mismatches == 0, f"Found {mismatches} target/winner mismatches"


def test_no_team1_equals_team2(df):
    """team1 and team2 must not be the same in any row."""
    same_teams = (df["team1"] == df["team2"]).sum()
    assert same_teams == 0, f"Found {same_teams} rows where team1 == team2"


def test_data_quality_report_runs(df):
    """data_quality_report must return a dict with expected keys."""
    report = data_quality_report(df)
    assert isinstance(report, dict)
    assert "total_rows" in report
    assert report["total_rows"] == len(df)


def test_chronological_ordering():
    """Dataset loaded and sorted by date should be in ascending order."""
    if not PROCESSED_PATH.exists():
        pytest.skip("Processed dataset not found")
    df = load_processed_data(PROCESSED_PATH)
    df = df.sort_values("date")
    dates = df["date"].dropna().values
    # Each date must be >= previous
    for i in range(1, len(dates)):
        assert dates[i] >= dates[i - 1], "Dates are not in ascending order"
