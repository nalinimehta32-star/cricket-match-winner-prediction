"""
src/train_model.py
Cricket Match Winner Prediction — Model Training

Trains four models and compares them:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Gradient Boosting

Uses TimeSeriesSplit for validation (time-aware; no future leakage).
Saves the best pipeline to models/best_model.joblib.
Saves metrics to models/model_metrics.json.
Saves feature metadata to models/feature_metadata.json.

Usage:
    python src/train_model.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Allow imports from src/ when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_processed_data
from evaluate_model import evaluate_model, build_model_comparison_table
from feature_engineering import (
    get_categorical_features,
    get_feature_columns,
    get_numeric_features,
    prepare_features,
)
from preprocessing import build_preprocessor, chronological_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Return a dictionary of model name → estimator.

    How each algorithm works (beginner-friendly):
    - Logistic Regression: Finds a linear decision boundary; outputs
      probabilities directly via the sigmoid function. Simple, fast,
      interpretable.
    - Decision Tree: Splits data on feature thresholds until reaching
      leaf nodes. Very interpretable but prone to overfitting.
    - Random Forest: Builds many decision trees on random subsets of
      data and features, then averages predictions. Reduces overfitting.
    - Gradient Boosting: Builds trees sequentially, each correcting the
      errors of the previous. Often high accuracy but slower to train.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED, max_depth=8
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, random_state=RANDOM_SEED
        ),
    }


# ---------------------------------------------------------------------------
# Time-aware cross-validation and hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(name: str, estimator, X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Pipeline:
    """Tune a model using GridSearchCV with TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", estimator)])

    param_grids = {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
        "Decision Tree": {
            "clf__max_depth": [4, 6, 8, 10, None],
            "clf__min_samples_split": [2, 5, 10],
        },
        "Random Forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 8, 12],
            "clf__min_samples_split": [2, 5],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [100, 150],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
        },
    }

    grid = param_grids.get(name, {})

    if grid:
        logger.info("Tuning %s with GridSearchCV (TimeSeriesSplit n=5)...", name)
        gs = GridSearchCV(
            pipeline,
            grid,
            cv=tscv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        logger.info(
            "%s best params: %s, CV ROC-AUC: %.4f",
            name, gs.best_params_, gs.best_score_
        )
        return gs.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_all_models() -> dict:
    """
    Full training pipeline:
    1. Load processed data.
    2. Prepare features (chronological historical stats).
    3. Chronological train/test split (80/20).
    4. Tune each model with TimeSeriesSplit CV.
    5. Evaluate on the held-out test set.
    6. Select best model by ROC-AUC.
    7. Save best pipeline, metrics, and metadata.

    Returns
    -------
    dict with keys: results, best_model_name, best_pipeline,
                    train_df, test_df, feature_cols
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading processed data...")
    df = load_processed_data()
    logger.info("Loaded %d matches", len(df))

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Computing features...")
    df_feat = prepare_features(df)
    feature_cols = get_feature_columns()

    # ------------------------------------------------------------------
    # 3. Chronological split
    # ------------------------------------------------------------------
    train_df, test_df = chronological_split(df_feat, test_fraction=0.20)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    logger.info(
        "Train: %d matches | Test: %d matches",
        len(train_df), len(test_df)
    )

    # ------------------------------------------------------------------
    # 4. Build preprocessor (fit only on training data)
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor()

    # ------------------------------------------------------------------
    # 5. Train and evaluate each model
    # ------------------------------------------------------------------
    models = get_models()
    results = {}

    for name, estimator in models.items():
        logger.info("Training %s...", name)
        # Build a fresh preprocessor for each model to avoid cross-contamination
        fresh_preprocessor = build_preprocessor()
        pipeline = tune_model(name, estimator, X_train, y_train, fresh_preprocessor)
        metrics = evaluate_model(pipeline, X_test, y_test, model_name=name)
        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
        }
        logger.info(
            "%s — Accuracy: %.4f, ROC-AUC: %.4f",
            name,
            metrics.get("accuracy", 0),
            metrics.get("roc_auc", 0),
        )

    # ------------------------------------------------------------------
    # 6. Select best model by ROC-AUC on test set
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda n: results[n]["metrics"].get("roc_auc", 0))
    best_pipeline = results[best_name]["pipeline"]
    logger.info("Best model: %s", best_name)

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    # 7a. Save best pipeline
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info("Best pipeline saved to %s", model_path)

    # 7b. Save metrics for all models
    serialisable_results = {}
    for name, res in results.items():
        m = res["metrics"].copy()
        # Remove non-serialisable numpy arrays
        for key in ["confusion_matrix", "fpr", "tpr"]:
            if key in m and isinstance(m[key], np.ndarray):
                m[key] = m[key].tolist()
            elif key in m and m[key] is not None:
                try:
                    m[key] = list(m[key])
                except Exception:
                    m[key] = None
        serialisable_results[name] = m

    metrics_to_save = {
        "models": serialisable_results,
        "best_model": best_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_min": str(train_df["date"].min().date()),
        "train_date_max": str(train_df["date"].max().date()),
        "test_date_min": str(test_df["date"].min().date()),
        "test_date_max": str(test_df["date"].max().date()),
    }
    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    # 7c. Save feature metadata
    cat_cols = get_categorical_features()
    cat_values = {}
    for col in cat_cols:
        if col in train_df.columns:
            cat_values[col] = sorted(train_df[col].dropna().unique().tolist())

    metadata = {
        "feature_columns": feature_cols,
        "categorical_features": cat_cols,
        "numeric_features": get_numeric_features(),
        "categorical_values": cat_values,
        "toss_decision_options": sorted(
            train_df["toss_decision"].dropna().unique().tolist()
        ) if "toss_decision" in train_df.columns else ["bat", "field"],
        "all_teams": sorted(
            set(train_df["team1"].dropna().unique())
            | set(train_df["team2"].dropna().unique())
        ),
        "all_venues": sorted(train_df["venue"].dropna().unique().tolist()),
        "recent_form_window": 5,
        "default_win_rate": 0.5,
        "target_definition": "1 = team1 wins, 0 = team2 wins",
        "prediction_timing": "post-toss, pre-match",
        "random_seed": RANDOM_SEED,
    }
    metadata_path = MODELS_DIR / "feature_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    table = build_model_comparison_table(
        {n: r["metrics"] for n, r in results.items()}
    )
    print("\nMODEL COMPARISON TABLE")
    print("=" * 80)
    print(table.to_string(index=False))
    print(f"\nBest model: {best_name}")

    return {
        "results": results,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    train_all_models()
