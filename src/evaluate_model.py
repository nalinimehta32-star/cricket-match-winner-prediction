"""
src/evaluate_model.py
Cricket Match Winner Prediction — Model Evaluation

Computes all required metrics from actual model predictions.
Nothing is hard-coded — all values come from sklearn's metric functions.

Positive class = 1 (team1 wins)
Negative class = 0 (team2 wins)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a fitted sklearn Pipeline on the test set.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_test : pd.DataFrame
    y_test : pd.Series
    model_name : str

    Returns
    -------
    dict containing:
        model_name, accuracy, precision, recall, f1, roc_auc,
        confusion_matrix, classification_report, fpr, tpr, thresholds
    """
    y_pred = pipeline.predict(X_test)

    # Check if model supports probability estimates
    has_proba = hasattr(pipeline, "predict_proba")
    if has_proba:
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
            has_proba = False
    else:
        y_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Team2 wins", "Team1 wins"])

    roc_auc = None
    fpr = tpr = thresholds_roc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
        except Exception as exc:
            logger.warning("ROC-AUC calculation failed: %s", exc)

    result = {
        "model_name": model_name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "classification_report": cr,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds_roc": thresholds_roc,
        "has_proba": has_proba,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "y_test": np.array(y_test),
    }

    return result


def build_model_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a DataFrame comparing all models' metrics.

    Parameters
    ----------
    results : dict
        {model_name: metrics_dict} from evaluate_model()

    Returns
    -------
    pd.DataFrame sorted by ROC-AUC descending.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy", None),
            "Precision": m.get("precision", None),
            "Recall": m.get("recall", None),
            "F1": m.get("f1", None),
            "ROC-AUC": m.get("roc_auc", None),
        })
    table = pd.DataFrame(rows)
    table = table.sort_values("ROC-AUC", ascending=False, na_position="last")
    return table


def print_metrics(metrics: dict) -> None:
    """Print a formatted metrics summary."""
    name = metrics.get("model_name", "Unknown")
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(f"Accuracy:   {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Precision:  {metrics.get('precision', 'N/A'):.4f}")
    print(f"Recall:     {metrics.get('recall', 'N/A'):.4f}")
    print(f"F1 Score:   {metrics.get('f1', 'N/A'):.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics.get('confusion_matrix')}")
    print(f"\nClassification Report:\n{metrics.get('classification_report')}")
