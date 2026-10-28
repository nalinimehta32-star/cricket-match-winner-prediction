"""
src/data_loader.py
Cricket Match Winner Prediction — Data Loading and Validation

Loads the processed T20I match dataset and performs comprehensive
data quality checks. All counts and statistics are derived from
the actual dataset — nothing is hard-coded.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"


# ---------------------------------------------------------------------------
# Primary loader
# ---------------------------------------------------------------------------

def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed T20I match dataset.

    Parameters
    ----------
    path : Path
        Path to the processed CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with parsed dates.

    Raises
    ------
    FileNotFoundError
        If the processed CSV does not exist.
    ValueError
        If the CSV cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}.\n"
            "Please run the data acquisition script first:\n"
            "  python src/feature_engineering.py\n"
            "Or follow the instructions in data/README.md to download "
            "the Cricsheet T20I archive."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc

    # Parse date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of schema problems (empty list means OK)."""
    required_cols = [
        "match_id", "date", "season", "team1", "team2",
        "venue", "toss_winner", "toss_decision", "winner", "target",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def check_target_classes(df: pd.DataFrame) -> dict:
    """Verify target column contains only 0 and 1."""
    counts = df["target"].value_counts().to_dict() if "target" in df.columns else {}
    invalid = df[~df["target"].isin([0, 1])]["target"].unique().tolist() if "target" in df.columns else []
    return {"class_counts": counts, "invalid_values": invalid}


def check_duplicates(df: pd.DataFrame) -> dict:
    """Check for duplicate match_id values."""
    if "match_id" not in df.columns:
        return {"duplicate_ids": 0, "duplicate_rows": 0}
    dup_ids = df["match_id"].duplicated().sum()
    dup_rows = df.duplicated().sum()
    return {"duplicate_ids": int(dup_ids), "duplicate_rows": int(dup_rows)}


def check_missing_values(df: pd.DataFrame) -> dict:
    """Return count of missing values per column."""
    return df.isnull().sum().to_dict()


def check_toss_validity(df: pd.DataFrame) -> dict:
    """Check that toss_winner is always one of team1/team2."""
    if not all(c in df.columns for c in ["team1", "team2", "toss_winner"]):
        return {"invalid_toss_winner": "columns missing"}
    invalid = df[
        (~df["toss_winner"].isna())
        & (df["toss_winner"] != df["team1"])
        & (df["toss_winner"] != df["team2"])
    ]
    return {"invalid_toss_winner_count": len(invalid)}


def check_winner_validity(df: pd.DataFrame) -> dict:
    """Check that winner is always one of team1/team2."""
    if not all(c in df.columns for c in ["team1", "team2", "winner"]):
        return {"invalid_winner": "columns missing"}
    invalid = df[
        (~df["winner"].isna())
        & (df["winner"] != df["team1"])
        & (df["winner"] != df["team2"])
    ]
    return {"invalid_winner_count": len(invalid)}


# ---------------------------------------------------------------------------
# Full data-quality report
# ---------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data-quality summary.

    All counts are derived from the actual DataFrame — nothing invented.
    """
    report = {}

    report["total_rows"] = len(df)

    # Schema
    schema_issues = validate_schema(df)
    report["schema_issues"] = schema_issues

    # Duplicates
    report["duplicates"] = check_duplicates(df)

    # Missing values
    report["missing_values"] = check_missing_values(df)

    # Date parsing
    if "date" in df.columns:
        bad_dates = df["date"].isna().sum()
        report["bad_dates"] = int(bad_dates)
        valid_dates = df["date"].notna()
        if valid_dates.any():
            report["date_range"] = {
                "min": str(df.loc[valid_dates, "date"].min().date()),
                "max": str(df.loc[valid_dates, "date"].max().date()),
            }

    # Target classes
    report["target_classes"] = check_target_classes(df)

    # Toss validity
    report["toss_validity"] = check_toss_validity(df)

    # Winner validity
    report["winner_validity"] = check_winner_validity(df)

    # Toss decision categories
    if "toss_decision" in df.columns:
        report["toss_decision_values"] = df["toss_decision"].value_counts().to_dict()

    # Teams
    if "team1" in df.columns and "team2" in df.columns:
        all_teams = set(df["team1"].dropna()) | set(df["team2"].dropna())
        report["unique_teams"] = sorted(all_teams)
        report["team_count"] = len(all_teams)

    # Venues
    if "venue" in df.columns:
        report["unique_venues"] = int(df["venue"].nunique())

    # Seasons
    if "season" in df.columns:
        report["seasons"] = sorted(df["season"].dropna().unique().tolist())

    # Class distribution
    if "target" in df.columns:
        counts = df["target"].value_counts().to_dict()
        total = len(df)
        report["class_distribution"] = {
            "team1_wins (target=1)": {
                "count": int(counts.get(1, 0)),
                "pct": round(counts.get(1, 0) / total * 100, 1),
            },
            "team2_wins (target=0)": {
                "count": int(counts.get(0, 0)),
                "pct": round(counts.get(0, 0) / total * 100, 1),
            },
        }

    return report


def print_quality_report(report: dict) -> None:
    """Pretty-print the data-quality report."""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Total rows:          {report.get('total_rows', 'N/A')}")

    if "date_range" in report:
        dr = report["date_range"]
        print(f"Date range:          {dr['min']} → {dr['max']}")

    if "duplicates" in report:
        d = report["duplicates"]
        print(f"Duplicate IDs:       {d.get('duplicate_ids', 0)}")
        print(f"Duplicate rows:      {d.get('duplicate_rows', 0)}")

    if "bad_dates" in report:
        print(f"Unparseable dates:   {report['bad_dates']}")

    if "team_count" in report:
        print(f"Unique teams:        {report['team_count']}")

    if "unique_venues" in report:
        print(f"Unique venues:       {report['unique_venues']}")

    if "schema_issues" in report and report["schema_issues"]:
        print(f"Schema issues:       {report['schema_issues']}")
    else:
        print("Schema:              OK")

    if "target_classes" in report:
        tc = report["target_classes"]
        print(f"Target classes:      {tc.get('class_counts', {})}")
        if tc.get("invalid_values"):
            print(f"  INVALID targets:   {tc['invalid_values']}")

    if "class_distribution" in report:
        print("\nClass distribution:")
        for label, info in report["class_distribution"].items():
            print(f"  {label}: {info['count']} ({info['pct']}%)")

    if "toss_decision_values" in report:
        print(f"\nToss decisions: {report['toss_decision_values']}")

    if "toss_validity" in report:
        tv = report["toss_validity"]
        invalid_ct = tv.get("invalid_toss_winner_count", "N/A")
        print(f"Invalid toss winners: {invalid_ct}")

    if "winner_validity" in report:
        wv = report["winner_validity"]
        invalid_ct = wv.get("invalid_winner_count", "N/A")
        print(f"Invalid winners:     {invalid_ct}")

    missing = report.get("missing_values", {})
    if any(v > 0 for v in missing.values()):
        print("\nMissing values:")
        for col, cnt in missing.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    df = load_processed_data()
    report = data_quality_report(df)
    print_quality_report(report)
