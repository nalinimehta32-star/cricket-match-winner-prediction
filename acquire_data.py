"""
acquire_data.py
Cricket Match Winner Prediction — Data Acquisition Script

Downloads the Cricsheet Men's T20I CSV archive (csv2 format), extracts
match-level records from the *_info.csv files, and creates:
    data/processed/t20i_matches.csv

Data source: https://cricsheet.org/downloads/
License:     Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
Attribution: Data provided by Cricsheet (https://cricsheet.org)

Usage:
    python acquire_data.py

Cricsheet csv2 info format (per-match _info.csv):
    version,<version>
    info,<key>,<value>
    info,<key>,<value>
    ...
Keys of interest: team, date, venue, toss_winner, toss_decision, winner,
                  match_type, season, match_id, result (for no-results)
"""

import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed. Run:\n  pip install requests")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

T20I_CSV_URL = "https://cricsheet.org/downloads/t20s_male_csv2.zip"
RAW_ZIP_PATH = RAW_DIR / "t20s_male_csv2.zip"
PROCESSED_CSV_PATH = PROCESSED_DIR / "t20i_matches.csv"

MANUAL_INSTRUCTIONS = f"""
==============================================================
MANUAL DATASET DOWNLOAD INSTRUCTIONS
==============================================================
The automatic download failed (network not available).

Please follow these steps:

1. Open your browser and go to:
   https://cricsheet.org/downloads/

2. Under "T20 Internationals (male)" — download the CSV format zip:
     t20s_male_csv2.zip

3. Place the zip file in:
   {RAW_DIR}

4. Re-run this script:
   python acquire_data.py

License: CC BY-SA 4.0 | Attribution: Cricsheet (https://cricsheet.org)
==============================================================
"""


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset() -> bool:
    if RAW_ZIP_PATH.exists() and RAW_ZIP_PATH.stat().st_size > 1_000_000:
        logger.info("Archive already exists (%s). Skipping download.",
                    RAW_ZIP_PATH.name)
        return True
    logger.info("Downloading: %s", T20I_CSV_URL)
    try:
        r = requests.get(T20I_CSV_URL, timeout=120, stream=True)
        r.raise_for_status()
        downloaded = 0
        with open(RAW_ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                downloaded += len(chunk)
        logger.info("Downloaded %.1f MB to %s", downloaded / 1_048_576, RAW_ZIP_PATH)
        return True
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if RAW_ZIP_PATH.exists():
            RAW_ZIP_PATH.unlink()
        return False


# ---------------------------------------------------------------------------
# Parse a single _info.csv content string
# ---------------------------------------------------------------------------

def parse_info_file(content: str, match_id: str) -> dict:
    """
    Parse the raw text of a Cricsheet *_info.csv file.

    Format:
        version,2.3.0
        info,key,value
        info,key,value,...

    Some rows have more than 3 comma-separated parts (e.g. registry rows).
    We only care about rows starting with 'info'.

    Teams appear as:  info,team,Australia
                      info,team,Sri Lanka
    Date:             info,date,2017/02/17
    """
    meta = {"match_id": match_id}
    teams = []
    dates = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 2)   # max 3 parts: type, key, value
        if len(parts) < 3:
            continue
        row_type, key, value = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if row_type != "info":
            continue

        # Multi-value fields
        if key == "team":
            teams.append(value)
        elif key == "date":
            dates.append(value)
        elif key in meta:
            pass  # already stored (first occurrence wins for single-value fields)
        else:
            meta[key] = value

    # Assign teams (Cricsheet lists them in home-away order for some matches,
    # but for T20Is there's no strict home/away; we take order as-is)
    if len(teams) >= 2:
        meta["team1"] = teams[0]
        meta["team2"] = teams[1]
    elif len(teams) == 1:
        meta["team1"] = teams[0]

    # Use the first (or only) date
    if dates:
        meta["date"] = dates[0]

    return meta


# ---------------------------------------------------------------------------
# Build one clean match record from parsed metadata
# ---------------------------------------------------------------------------

def build_record(meta: dict) -> dict | None:
    """
    Convert raw parsed metadata to a standardised match record.
    Returns None for matches to be excluded.
    """
    team1 = meta.get("team1", "").strip()
    team2 = meta.get("team2", "").strip()
    winner = meta.get("winner", "").strip()
    match_type = meta.get("match_type", "").strip()
    outcome = meta.get("outcome", "").strip()      # older format
    result = meta.get("result", "").strip()        # sometimes present

    # Must be T20 format
    if match_type.upper() not in ("T20", "T20I"):
        return None

    # Must have both teams
    if not team1 or not team2:
        return None

    # Excluded outcomes
    no_result_markers = {"no result", "tie", "tied", "nan", ""}
    if winner.lower() in no_result_markers:
        return None
    if outcome.lower() in no_result_markers - {""}:
        return None
    if result.lower() in ("no result", "abandoned"):
        return None

    # Winner must be one of the two teams
    if winner not in (team1, team2):
        return None

    # Binary target
    target = 1 if winner == team1 else 0

    # Parse date (Cricsheet uses YYYY/MM/DD)
    date_str = meta.get("date", "")
    date_parsed = pd.to_datetime(date_str, errors="coerce", format="%Y/%m/%d")
    if date_parsed is pd.NaT or pd.isna(date_parsed):
        # Try alternative formats
        date_parsed = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(date_parsed):
        return None

    # Season
    season = meta.get("season", str(date_parsed.year))

    # Toss decision normalisation: Cricsheet uses 'bat' or 'field'
    toss_decision = meta.get("toss_decision", "").strip().lower()

    return {
        "match_id": meta.get("match_id", ""),
        "date": date_parsed.strftime("%Y-%m-%d"),
        "season": season,
        "team1": team1,
        "team2": team2,
        "venue": meta.get("venue", "").strip(),
        "toss_winner": meta.get("toss_winner", "").strip(),
        "toss_decision": toss_decision,
        "winner": winner,
        "match_type": match_type,
        "result_type": result or outcome,
        "target": target,
    }


# ---------------------------------------------------------------------------
# Process the zip archive
# ---------------------------------------------------------------------------

def process_archive(zip_path: Path) -> tuple[pd.DataFrame, dict]:
    logger.info("Processing archive: %s", zip_path)

    excluded = {
        "no_result": 0,
        "tie": 0,
        "wrong_format": 0,
        "unknown_winner": 0,
        "missing_teams": 0,
        "bad_date": 0,
        "parse_error": 0,
    }
    records = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        info_files = [f for f in all_files if f.endswith("_info.csv")]
        logger.info("Found %d info files", len(info_files))

        for fname in info_files:
            match_id = Path(fname).stem.replace("_info", "")
            try:
                with zf.open(fname) as f:
                    content = f.read().decode("utf-8", errors="replace")
                meta = parse_info_file(content, match_id)
                record = build_record(meta)

                if record is None:
                    # Categorise exclusion reason
                    team1 = meta.get("team1", "")
                    team2 = meta.get("team2", "")
                    winner = meta.get("winner", "")
                    match_type = meta.get("match_type", "").upper()
                    date_str = meta.get("date", "")

                    if match_type not in ("T20", "T20I"):
                        excluded["wrong_format"] += 1
                    elif not team1 or not team2:
                        excluded["missing_teams"] += 1
                    elif winner.lower() in ("tie", "tied"):
                        excluded["tie"] += 1
                    elif winner.lower() in ("no result", ""):
                        excluded["no_result"] += 1
                    elif not pd.isna(pd.to_datetime(date_str, errors="coerce")):
                        if winner and winner not in (team1, team2):
                            excluded["unknown_winner"] += 1
                        else:
                            excluded["no_result"] += 1
                    else:
                        excluded["bad_date"] += 1
                else:
                    records.append(record)

            except Exception as exc:
                excluded["parse_error"] += 1
                logger.debug("Error parsing %s: %s", fname, exc)

    logger.info("Valid records: %d", len(records))
    logger.info("Excluded: %s", excluded)

    if not records:
        return pd.DataFrame(), excluded

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Remove exact duplicate match_ids
    before = len(df)
    df = df.drop_duplicates(subset=["match_id"])
    if len(df) < before:
        logger.info("Removed %d duplicate match IDs", before - len(df))

    return df, excluded


# ---------------------------------------------------------------------------
# Write data/README.md
# ---------------------------------------------------------------------------

def write_data_readme(df: pd.DataFrame, excluded: dict) -> None:
    valid_matches = len(df) if df is not None and not df.empty else "N/A"
    date_min = str(df["date"].min().date()) if valid_matches != "N/A" else "N/A"
    date_max = str(df["date"].max().date()) if valid_matches != "N/A" else "N/A"
    unique_teams = len(set(df["team1"].dropna()) | set(df["team2"].dropna())) if valid_matches != "N/A" else "N/A"

    readme = f"""# Data Directory — Cricket Match Winner Prediction

## Source

- **Provider:** Cricsheet (https://cricsheet.org)
- **Dataset:** Men's T20 International (T20I) matches
- **Download URL:** {T20I_CSV_URL}
- **Format:** Cricsheet csv2 — per-match `*_info.csv` files inside zip archive
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Attribution:** Data provided by Cricsheet (https://cricsheet.org)

## Directory Structure

```
data/
├── raw/
│   └── t20s_male_csv2.zip     # Original archive — NOT committed to git (large file)
├── processed/
│   └── t20i_matches.csv       # One-row-per-match processed dataset
└── README.md                  # This file
```

## Redistribution Note

The raw data is licensed CC BY-SA 4.0 (redistribution with attribution permitted).
The raw zip is excluded from this repository via `.gitignore` due to its size.
Reproduction instructions are provided instead.

## Processed Dataset Schema (`t20i_matches.csv`)

| Column | Type | Description |
|--------|------|-------------|
| match_id | string | Cricsheet match identifier (traceability only — not used as feature) |
| date | date | Match date (YYYY-MM-DD) |
| season | string | Season label (e.g. 2016/17) or calendar year |
| team1 | string | Reference team (target=1 if team1 wins) |
| team2 | string | Opposing team |
| venue | string | Ground name |
| toss_winner | string | Team that won the toss |
| toss_decision | string | 'bat' or 'field' |
| winner | string | Official match winner |
| match_type | string | 'T20' or 'T20I' |
| result_type | string | Additional result info (e.g. 'by 5 wickets') |
| target | int | Binary: 1 = team1 won, 0 = team2 won |

## Valid Matches

- **Total valid matches:** {valid_matches}
- **Date range:** {date_min} to {date_max}
- **Unique teams:** {unique_teams}

## Exclusion Summary

| Reason | Count |
|--------|-------|
| No result / abandoned | {excluded.get('no_result', 'N/A')} |
| Tied matches | {excluded.get('tie', 'N/A')} |
| Wrong format (non-T20) | {excluded.get('wrong_format', 'N/A')} |
| Unknown winner | {excluded.get('unknown_winner', 'N/A')} |
| Missing team names | {excluded.get('missing_teams', 'N/A')} |
| Unparseable date | {excluded.get('bad_date', 'N/A')} |
| Parse errors | {excluded.get('parse_error', 'N/A')} |

## Processing Decisions

1. Only `*_info.csv` files were read (not ball-by-ball delivery files).
2. Matches with no result, ties, or unknown winners are excluded.
3. `team1` / `team2` order follows the order Cricsheet lists teams (arbitrary).
4. `match_id` is preserved for traceability but is **not** used as a feature.
5. `target = 1` if `team1` won; `target = 0` if `team2` won.
6. Default win rate 0.5 is used as prior for teams with no history.

## Known Limitations

- Venue names may vary across eras for the same ground.
- Older matches may have incomplete metadata.
- Team naming follows Cricsheet conventions which may differ from popular usage.

## Reproduction

```bash
# Option 1: Automatic
python acquire_data.py

# Option 2: Manual
# 1. Download t20s_male_csv2.zip from https://cricsheet.org/downloads/
# 2. Place in data/raw/
# 3. python acquire_data.py
```

## Attribution

> Data provided by Cricsheet (https://cricsheet.org) under the
> Creative Commons Attribution-ShareAlike 4.0 International License.
"""
    readme_path = PROJECT_ROOT / "data" / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    logger.info("data/README.md written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Download
    if not download_dataset():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    # 2. Process
    df, excluded = process_archive(RAW_ZIP_PATH)

    if df.empty:
        logger.error("No valid match records extracted.")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    # 3. Save processed CSV
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    logger.info("Saved processed CSV: %s (%d rows)", PROCESSED_CSV_PATH, len(df))

    # 4. Write README
    write_data_readme(df, excluded)

    # 5. Summary
    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 60)
    print(f"Valid matches:   {len(df)}")
    print(f"Date range:      {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Unique teams:    {len(set(df['team1']) | set(df['team2']))}")
    print(f"Unique venues:   {df['venue'].nunique()}")
    print(f"Class balance:")
    print(f"  Team1 wins:    {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
    print(f"  Team2 wins:    {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    print(f"Excluded:")
    for k, v in excluded.items():
        if v:
            print(f"  {k}: {v}")
    print(f"Saved to:        {PROCESSED_CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
