# Cricket Match Winner Prediction Using Machine Learning

A complete Machine Learning project that predicts the winner of Men's T20 International (T20I) cricket matches using historical data from [Cricsheet](https://cricsheet.org).

## Project Overview

This project builds a supervised binary classification model to predict the outcome of a Men's T20I cricket match. The model is trained on real historical match data and deployed as an interactive Streamlit web application.

**Prediction timing:** Post-toss, pre-match — the model uses information available after the toss but before the first ball.

> **Disclaimer:** Predictions are estimates based on historical data and do not guarantee match outcomes. Cricket outcomes depend on many factors including team selection, player availability, weather and match conditions. This project is for educational purposes only.

---

## Problem Statement

Given match conditions known after the toss (team identities, venue, toss winner, toss decision) and historical team performance statistics, can we predict which team is more likely to win a T20I match?

**Target variable:** `1` = Team 1 wins, `0` = Team 2 wins (binary classification)

---

## Objectives

1. Acquire real historical T20I match data (Cricsheet)
2. Build meaningful, leak-free features from historical records
3. Train and compare multiple ML algorithms with chronological validation
4. Select the best model based on evidence
5. Deploy an interactive prediction app using Streamlit
6. Document everything for reproducibility

---

## Dataset

- **Source:** [Cricsheet](https://cricsheet.org) — created by Stephen Rushe
- **URL:** https://cricsheet.org/downloads/ (Men's T20I, CSV format)
- **File:** `t20s_male_csv2.zip`
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Attribution:** Data provided by Cricsheet (https://cricsheet.org)

### Dataset Statistics (Actual)

| Metric | Value |
|--------|-------|
| Valid matches | 3,527 |
| Date range | 2005-02-17 to 2026-10-25 |
| Unique teams | 109 |
| Unique venues | 337 |
| Team 1 wins (target=1) | 1,725 (48.9%) |
| Team 2 wins (target=0) | 1,802 (51.1%) |

### Dataset Schema

| Column | Description |
|--------|-------------|
| match_id | Cricsheet identifier (traceability only) |
| date | Match date (YYYY-MM-DD) |
| season | Season label |
| team1 | Reference team (target=1 if wins) |
| team2 | Opposing team |
| venue | Ground name |
| toss_winner | Team that won the toss |
| toss_decision | bat or field |
| winner | Official match winner |
| match_type | T20 or T20I |
| result_type | Additional result info |
| target | Binary: 1=team1 won, 0=team2 won |

---

## Feature Engineering

All historical features are computed using **only matches that occurred before** the match being predicted. No future information is used.

| Feature | Description |
|---------|-------------|
| team1 | Team 1 identity (categorical) |
| team2 | Team 2 identity (categorical) |
| venue | Match venue (categorical) |
| toss_winner_enc | Toss winner: "team1" or "team2" |
| toss_decision | bat or field |
| match_year | Calendar year |
| team1_win_rate | Team 1 historical win rate (all prior matches) |
| team2_win_rate | Team 2 historical win rate |
| team1_recent_form | Team 1 win rate in last 5 matches |
| team2_recent_form | Team 2 win rate in last 5 matches |
| h2h_team1_win_rate | Head-to-head win rate for Team 1 vs Team 2 |

**Prior for teams with no history:** 0.5 (neutral, documented fallback)

---

## ML Algorithms

| Algorithm | Description |
|-----------|-------------|
| Logistic Regression | Linear classifier with sigmoid probability output |
| Decision Tree | Tree of feature-threshold rules |
| Random Forest | Ensemble of 100–200 decision trees |
| Gradient Boosting | Sequential error-correcting ensemble |

---

## Validation Methodology

- **Train/Test Split:** Chronological — earliest 80% for training, latest 20% for testing
- **Cross-Validation:** TimeSeriesSplit (5 folds) for hyperparameter tuning
- **No data leakage:** Future match outcomes never influence earlier predictions
- **Hyperparameter tuning:** GridSearchCV with TimeSeriesSplit, optimising ROC-AUC

---

## Model Evaluation (Actual Results)

All metrics computed on the **chronological test set (706 matches, 2025-09-12 to 2026-10-25)**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|-----------|--------|-----|---------|
| **Gradient Boosting** ★ | **0.7082** | **0.7168** | **0.6884** | **0.7023** | **0.7889** |
| Random Forest | 0.7139 | 0.7352 | 0.6686 | 0.7003 | 0.7877 |
| Logistic Regression | 0.6983 | 0.7201 | 0.6487 | 0.6826 | 0.7844 |
| Decision Tree | 0.6856 | 0.6866 | 0.6827 | 0.6847 | 0.7377 |

★ = Selected best model (highest ROC-AUC)

### Train/Test Split Details

- **Train:** 2,821 matches (2005-02-17 to 2025-09-12)
- **Test:** 706 matches (2025-09-12 to 2026-10-25)

---

## Final Model

**Gradient Boosting Classifier** was selected as the best model based on highest ROC-AUC (0.7889) on the chronological test set.

Selection rationale:
- Highest ROC-AUC among all four models
- Good balance of precision and recall
- Supports probability estimates (needed for the prediction UI)

---

## Streamlit Application Features

1. **Home** — Project overview, dataset statistics, workflow
2. **Match Prediction** — Interactive form; post-toss match winner prediction
3. **Historical Analytics** — Team stats, venue distribution, toss analysis, head-to-head
4. **Model Performance** — Model comparison table, confusion matrix, ROC curve, feature importance
5. **About Project** — Full methodology, feature engineering, technologies, limitations

---

## Project Structure

```
cricket-match-winner-prediction/
├── data/
│   ├── raw/                        # Original Cricsheet archive (not in git)
│   ├── processed/
│   │   └── t20i_matches.csv        # One-row-per-match dataset
│   └── README.md                   # Data documentation
├── models/
│   ├── best_model.joblib           # Saved best pipeline
│   ├── model_metrics.json          # Actual evaluation metrics
│   └── feature_metadata.json       # Feature definitions and categories
├── notebooks/
│   └── cricket_analysis.ipynb      # Exploratory data analysis
├── outputs/
│   └── figures/                    # All generated charts
│       ├── matches_by_year.png
│       ├── team_win_counts.png
│       ├── venue_distribution.png
│       ├── toss_analysis.png
│       ├── class_distribution.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load and validate data
│   ├── preprocessing.py            # Scikit-learn preprocessing pipeline
│   ├── feature_engineering.py      # Historical feature computation
│   ├── train_model.py              # Model training and selection
│   ├── evaluate_model.py           # Metric calculation
│   └── predict.py                  # Reusable prediction function
├── tests/
│   ├── test_data.py                # Data validation tests
│   ├── test_model.py               # Model artifact tests
│   └── test_prediction.py          # Prediction function tests
├── screenshots/                    # App screenshots
├── app.py                          # Streamlit application
├── acquire_data.py                 # Dataset download + processing
├── generate_plots.py               # EDA plot generation
├── requirements.txt
├── pytest.ini
├── .gitignore
├── LICENSE
└── README.md
```

---

## Technologies Used

| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.11.9 |
| Data | pandas | 3.1.0 |
| Data | numpy | 2.4.6 |
| ML | scikit-learn | 1.9.1 |
| Visualisation | matplotlib | 3.11.2 |
| Visualisation | seaborn | 0.13.2 |
| Application | Streamlit | 1.66.1 |
| Serialisation | joblib | 1.6.0 |
| Testing | pytest | 9.2.0 |
| HTTP | requests | 2.34.2 |
| Version Control | Git | 2.56.0 |

---

## Installation

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/nalinimehta32-star/cricket-match-winner-prediction.git
cd cricket-match-winner-prediction

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## How to Download/Prepare the Dataset

```bash
# Automatic download from Cricsheet
python acquire_data.py
```

If automatic download fails (no internet), follow these manual steps:
1. Go to https://cricsheet.org/downloads/
2. Download "T20 Internationals (male)" → CSV format: `t20s_male_csv2.zip`
3. Place the zip in `data/raw/`
4. Run `python acquire_data.py`

---

## How to Train the Model

```bash
python src/train_model.py
```

This will:
1. Load and feature-engineer the dataset
2. Create an 80/20 chronological split
3. Tune all four models with TimeSeriesSplit cross-validation
4. Evaluate on the held-out test set
5. Save the best pipeline to `models/best_model.joblib`
6. Save metrics to `models/model_metrics.json`
7. Save metadata to `models/feature_metadata.json`

---

## How to Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How to Run Tests

```bash
python -m pytest tests/ -v
```

**Test results (verified):** 40/40 tests passed.

---

## Limitations

1. **No player-level data** — team selection, player form, injuries not included
2. **No pitch/weather data** — conditions heavily influence T20I outcomes
3. **Historical bias** — teams with more match history have more reliable statistics
4. **Venue naming** — the same ground may appear under different names
5. **Probability calibration** — probabilities are model estimates, not formally calibrated
6. **Temporal gap** — the model was trained on data through Oct 2026; new teams may have no history

---

## Future Scope

- Include player-level performance statistics
- Add weather and pitch condition data
- Ball-by-ball in-match live prediction
- Formal probability calibration
- Expand to ODI and Test formats
- REST API deployment

---

## Author

MBA Data Science and AI Student  
Built as a Machine Learning Mini-Project

---

## References

1. Cricsheet — https://cricsheet.org (Data source, CC BY-SA 4.0)
2. scikit-learn — https://scikit-learn.org
3. Streamlit — https://docs.streamlit.io
4. pandas — https://pandas.pydata.org
5. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
6. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189–1232.
