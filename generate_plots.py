"""
generate_plots.py
Cricket Match Winner Prediction — EDA Plot Generation

Generates all EDA charts and saves them to outputs/figures/.
Reads from data/processed/t20i_matches.csv.

Usage:
    python generate_plots.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "Set2"


def save_fig(name: str) -> None:
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def plot_matches_by_year(df: pd.DataFrame) -> None:
    """Bar chart: number of T20I matches per calendar year."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    counts = df.groupby("year").size().reset_index(name="matches")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(counts["year"].astype(str), counts["matches"], color=sns.color_palette(PALETTE, len(counts)))
    ax.set_title("Men's T20I Matches per Year", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Matches")
    ax.tick_params(axis="x", rotation=45)
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=8
        )
    plt.tight_layout()
    save_fig("matches_by_year.png")


def plot_team_win_counts(df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart: total wins per team."""
    team1_wins = df[df["target"] == 1]["team1"].value_counts()
    team2_wins = df[df["target"] == 0]["team2"].value_counts()
    total_wins = team1_wins.add(team2_wins, fill_value=0).sort_values(ascending=False)
    total_wins = total_wins.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(total_wins.index[::-1], total_wins.values[::-1],
                   color=sns.color_palette(PALETTE, len(total_wins)))
    ax.set_title(f"Top {top_n} Teams by Total T20I Wins", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Wins")
    for bar in bars:
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=9
        )
    plt.tight_layout()
    save_fig("team_win_counts.png")


def plot_venue_distribution(df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart: most used venues."""
    venue_counts = df["venue"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        venue_counts.index[::-1],
        venue_counts.values[::-1],
        color=sns.color_palette("Blues_r", len(venue_counts))
    )
    ax.set_title(f"Top {top_n} Venues by Match Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Matches")
    for bar in bars:
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())), va="center", fontsize=9
        )
    plt.tight_layout()
    save_fig("venue_distribution.png")


def plot_toss_analysis(df: pd.DataFrame) -> None:
    """Two-panel chart: toss decision distribution + toss winner vs match winner."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Toss decision
    if "toss_decision" in df.columns:
        toss_dec = df["toss_decision"].value_counts()
        axes[0].pie(
            toss_dec.values,
            labels=toss_dec.index,
            autopct="%1.1f%%",
            colors=sns.color_palette(PALETTE, len(toss_dec)),
            startangle=90
        )
        axes[0].set_title("Toss Decision Distribution", fontweight="bold")
    else:
        axes[0].text(0.5, 0.5, "Toss decision not available",
                     ha="center", va="center")

    # Panel 2: Toss winner vs match winner
    if "toss_winner" in df.columns:
        # Does the team that won the toss win the match?
        df = df.copy()
        df["toss_winner_won"] = (
            ((df["toss_winner"] == df["team1"]) & (df["target"] == 1))
            | ((df["toss_winner"] == df["team2"]) & (df["target"] == 0))
        )
        toss_win_rate = df["toss_winner_won"].mean() * 100
        labels = ["Toss winner won", "Toss winner lost"]
        sizes = [toss_win_rate, 100 - toss_win_rate]
        colors = ["#2ecc71", "#e74c3c"]
        axes[1].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        axes[1].set_title(
            f"Toss Winner Match Outcome\n"
            f"(Toss winner won {toss_win_rate:.1f}% of matches)",
            fontweight="bold"
        )
    else:
        axes[1].text(0.5, 0.5, "Toss data not available", ha="center", va="center")

    plt.suptitle(
        "Note: Winning the toss does not guarantee winning the match.",
        fontsize=9, style="italic", y=0.02
    )
    plt.tight_layout()
    save_fig("toss_analysis.png")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart: target class distribution."""
    if "target" not in df.columns:
        return

    counts = df["target"].value_counts()
    labels = {0: "Team 2 wins (0)", 1: "Team 1 wins (1)"}
    bar_labels = [labels.get(i, str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(bar_labels, counts.values, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    total = len(df)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10
        )
    plt.tight_layout()
    save_fig("class_distribution.png")


def plot_model_comparison(metrics_dict: dict) -> None:
    """Grouped bar chart comparing model metrics."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    models = list(metrics_dict.keys())
    x = np.arange(len(metric_cols))
    width = 0.8 / len(models)
    colors = sns.color_palette(PALETTE, len(models))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, metrics) in enumerate(metrics_dict.items()):
        vals = [
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
            metrics.get("roc_auc", 0) or 0,
        ]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[i])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Chronological Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_fig("model_comparison.png")


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Best Model") -> None:
    """Heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Team2 wins (0)", "Team1 wins (1)"],
        yticklabels=["Team2 wins (0)", "Team1 wins (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    save_fig("confusion_matrix.png")


def plot_roc_curve(fpr, tpr, roc_auc: float, model_name: str = "Best Model") -> None:
    """ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})", color="#2980b9")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2980b9")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    save_fig("roc_curve.png")


def plot_feature_importance(pipeline, feature_names: list, model_name: str = "Best Model") -> None:
    """Feature importance for tree-based models."""
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return
    if not hasattr(clf, "feature_importances_"):
        logger.info("Model %s does not expose feature_importances_", model_name)
        return

    importances = clf.feature_importances_
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None:
        try:
            feat_names = preprocessor.get_feature_names_out()
        except Exception:
            feat_names = [f"feat_{i}" for i in range(len(importances))]
    else:
        feat_names = feature_names

    # Sort descending
    idx = np.argsort(importances)[::-1][:20]
    top_names = [feat_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_vals[::-1], color=sns.color_palette("viridis", len(top_names)))
    ax.set_title(f"Feature Importance — {model_name} (Top 20)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    save_fig("feature_importance.png")


def generate_all_plots(df: pd.DataFrame) -> None:
    """Generate all EDA plots from the dataset."""
    logger.info("Generating EDA plots...")

    plot_matches_by_year(df)
    plot_team_win_counts(df)
    plot_venue_distribution(df)
    plot_toss_analysis(df)
    plot_class_distribution(df)

    logger.info("EDA plots saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    df = load_processed_data()
    generate_all_plots(df)
    print(f"\nAll EDA plots saved to: {FIGURES_DIR}")
