# Data Directory — Cricket Match Winner Prediction

## Source

- **Provider:** Cricsheet (https://cricsheet.org)
- **Dataset:** Men's T20 International (T20I) matches
- **Download URL:** https://cricsheet.org/downloads/t20s_male_csv2.zip
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

- **Total valid matches:** 3527
- **Date range:** 2005-02-17 to 2026-10-25
- **Unique teams:** 109

## Exclusion Summary

| Reason | Count |
|--------|-------|
| No result / abandoned | 119 |
| Tied matches | 0 |
| Wrong format (non-T20) | 0 |
| Unknown winner | 0 |
| Missing team names | 0 |
| Unparseable date | 0 |
| Parse errors | 0 |

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
