"""
app.py
Cricket Match Winner Prediction — Streamlit Application

Professional T20I cricket analytics dashboard with five sections:
  1. Home
  2. Match Prediction
  3. Historical Analytics
  4. Model Performance
  5. About Project

Usage:
    streamlit run app.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Match Winner Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "t20i_matches.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "feature_metadata.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> Optional[pd.DataFrame]:
    """Load the processed match dataset."""
    if not PROCESSED_DATA_PATH.exists():
        return None
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the trained sklearn Pipeline."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[dict]:
    """Load model metrics JSON."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Failed to load metrics: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_metadata() -> Optional[dict]:
    """Load feature metadata JSON."""
    if not METADATA_PATH.exists():
        return None
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Failed to load metadata: {exc}")
        return None


# ── Shared utility ────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def data_unavailable_warning() -> None:
    st.warning(
        "The processed dataset is not yet available. "
        "Please run the data acquisition script first:\n\n"
        "```\npython acquire_data.py\n```"
    )


def model_unavailable_warning() -> None:
    st.warning(
        "The trained model is not yet available. "
        "Please train the model first:\n\n"
        "```\npython src/train_model.py\n```"
    )


# ── Navigation ─────────────────────────────────────────────────────────────────

PAGES = [
    "Home",
    "Match Prediction",
    "Historical Analytics",
    "Model Performance",
    "About Project",
]

with st.sidebar:
    st.markdown("# 🏏 Cricket Predictor")
    st.markdown("**Men's T20I Match Winner Prediction**")
    st.markdown("---")
    selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Data: Cricsheet (CC BY-SA 4.0)")
    st.caption("For educational purposes only.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if selected_page == "Home":
    st.title("Cricket Match Winner Prediction")
    st.markdown("### Using Machine Learning on Historical T20I Data")
    st.markdown("---")

    df = load_data()
    metrics = load_metrics()
    metadata = load_metadata()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_matches = len(df) if df is not None else "N/A"
        st.metric("Total Matches", n_matches)
    with col2:
        if df is not None and not df.empty:
            teams = len(set(df["team1"].dropna()) | set(df["team2"].dropna()))
        else:
            teams = "N/A"
        st.metric("Teams Covered", teams)
    with col3:
        if df is not None and not df.empty:
            date_range = f"{df['date'].min().year}–{df['date'].max().year}"
        else:
            date_range = "N/A"
        st.metric("Period", date_range)
    with col4:
        best_model = metrics.get("best_model", "N/A") if metrics else "N/A"
        st.metric("Best Model", best_model)

    st.markdown("---")

    # Overview columns
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### Project Objective")
        st.markdown(
            """
            This project predicts the winner of a **Men's T20 International (T20I)** cricket match
            using Machine Learning. Given match conditions known **after the toss but before the
            first ball is bowled**, the model estimates the probability that each team wins.

            **This is a post-toss, pre-match predictor.**
            Toss details (winner and decision) are included as inputs because they are
            known after the toss but before play begins.

            > ⚠️ Predictions are model estimates based on historical patterns. They are not
            > guaranteed to be correct. Cricket outcomes depend on many factors including
            > team selection, player form, weather, and match conditions.
            """
        )

        st.markdown("### Dataset")
        st.markdown(
            f"""
            - **Source:** [Cricsheet](https://cricsheet.org) — Men's T20I matches
            - **License:** Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
            - **Format:** One row per match
            - **Matches:** {n_matches}
            - **Period:** {date_range}
            """
        )

        st.markdown("### ML Algorithms Compared")
        st.markdown(
            """
            1. **Logistic Regression** — linear model with probability output
            2. **Decision Tree** — rule-based tree structure
            3. **Random Forest** — ensemble of decision trees
            4. **Gradient Boosting** — sequential error-correcting ensemble
            """
        )

    with right:
        st.markdown("### Project Workflow")
        st.markdown(
            """
            ```
            1. Data Acquisition (Cricsheet)
                 ↓
            2. Data Cleaning & Validation
                 ↓
            3. Feature Engineering
               (historical win rates, recent form,
                head-to-head — no future leakage)
                 ↓
            4. Chronological Train/Test Split (80/20)
                 ↓
            5. Model Training (4 algorithms)
                 ↓
            6. Cross-Validation (TimeSeriesSplit)
                 ↓
            7. Model Comparison & Selection
                 ↓
            8. Pipeline Saved → Streamlit App
            ```
            """
        )

        st.markdown("### Features Used")
        st.markdown(
            """
            - Team 1 & Team 2 identities
            - Venue
            - Toss winner (relative: team1 or team2)
            - Toss decision (bat/field)
            - Match year
            - Team 1 & 2 historical win rate
            - Team 1 & 2 recent form (last 5 matches)
            - Head-to-head win rate
            """
        )

    st.markdown("---")
    st.markdown("### Limitations")
    st.markdown(
        """
        - Predictions are estimates based on historical data only.
        - Player availability, injuries, and current form are not captured.
        - Weather and pitch conditions are not included.
        - The model cannot account for team-selection changes on match day.
        - Historical win rates for newer or infrequent teams are based on limited data.
        - This project is for **educational purposes only** and is not intended
          for betting or financial decisions.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif selected_page == "Match Prediction":
    section_header(
        "Match Prediction",
        "Post-toss, pre-match prediction. Enter match conditions after the toss."
    )

    df = load_data()
    pipeline = load_pipeline()
    metadata = load_metadata()

    if df is None:
        data_unavailable_warning()
        st.stop()
    if pipeline is None:
        model_unavailable_warning()
        st.stop()
    if metadata is None:
        st.error("Feature metadata not found. Please re-run training.")
        st.stop()

    # Get available teams and venues from metadata
    all_teams = metadata.get("all_teams", sorted(set(df["team1"].dropna()) | set(df["team2"].dropna())))
    all_venues = metadata.get("all_venues", sorted(df["venue"].dropna().unique().tolist()))
    toss_decisions = metadata.get("toss_decision_options", ["bat", "field"])

    st.markdown("#### Match Details")
    st.caption("All fields below should reflect information available after the toss.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1 (Reference Team)", all_teams, index=0)
    with col2:
        team2_options = [t for t in all_teams if t != team1]
        team2 = st.selectbox("Team 2", team2_options, index=0)

    col3, col4 = st.columns(2)
    with col3:
        venue = st.selectbox("Venue", all_venues)
    with col4:
        prediction_date = st.date_input(
            "Prediction Date",
            value=pd.Timestamp.today().date(),
            help="Date of the match (used to compute historical features)"
        )

    col5, col6 = st.columns(2)
    with col5:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col6:
        toss_decision = st.selectbox("Toss Decision", toss_decisions)

    st.markdown("---")
    predict_button = st.button("Predict Match Winner", type="primary", use_container_width=True)

    if predict_button:
        # Validate inputs
        errors = []
        if team1 == team2:
            errors.append("Team 1 and Team 2 must be different.")
        if toss_winner not in (team1, team2):
            errors.append("Toss winner must be one of the two selected teams.")
        if toss_decision not in toss_decisions:
            errors.append(f"Toss decision must be one of: {toss_decisions}")

        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Computing prediction..."):
                try:
                    from predict import predict_match
                    result = predict_match(
                        team1=team1,
                        team2=team2,
                        venue=venue,
                        toss_winner=toss_winner,
                        toss_decision=toss_decision,
                        prediction_date=pd.Timestamp(prediction_date),
                        pipeline=pipeline,
                        metadata=metadata,
                        historical_df=df,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

            if result.get("errors"):
                for err in result["errors"]:
                    st.error(err)
            else:
                # Display results
                st.markdown("---")
                st.markdown("### Prediction Results")

                winner = result["predicted_winner"]
                t1_prob = result.get("team1_win_prob")
                t2_prob = result.get("team2_win_prob")

                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Team 1", team1)
                with res_col2:
                    st.metric("Predicted Winner", f"🏆 {winner}")
                with res_col3:
                    st.metric("Team 2", team2)

                if t1_prob is not None and t2_prob is not None:
                    st.markdown("#### Model-Estimated Win Probabilities")
                    st.caption(
                        "These are model estimates, not guaranteed probabilities. "
                        "The model has not been formally calibrated."
                    )

                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.markdown(f"**{team1}**")
                        st.progress(float(t1_prob), text=f"{t1_prob*100:.1f}%")
                    with prob_col2:
                        st.markdown(f"**{team2}**")
                        st.progress(float(t2_prob), text=f"{t2_prob*100:.1f}%")

                    # Simple bar chart
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7, 3))
                    bars = ax.barh(
                        [team2, team1],
                        [t2_prob * 100, t1_prob * 100],
                        color=["#e74c3c" if team2 != winner else "#2ecc71",
                               "#2ecc71" if team1 == winner else "#e74c3c"],
                        edgecolor="white"
                    )
                    for bar in bars:
                        ax.text(
                            bar.get_width() + 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{bar.get_width():.1f}%",
                            va="center", fontweight="bold"
                        )
                    ax.set_xlim(0, 110)
                    ax.set_xlabel("Estimated Win Probability (%)")
                    ax.set_title("Model-Estimated Win Probabilities", fontweight="bold")
                    ax.axvline(50, color="gray", linestyle="--", lw=1, alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Features used
                with st.expander("Features used for this prediction"):
                    features_used = result.get("features_used", {})
                    if features_used:
                        feat_df = pd.DataFrame([features_used]).T
                        feat_df.columns = ["Value"]
                        st.dataframe(feat_df)

                # Disclaimer
                st.info(
                    "Predictions are based on historical data and may be incorrect. "
                    "Cricket outcomes depend on many factors, including team selection, "
                    "player availability, weather and match conditions. "
                    "**This project is for educational purposes only.**"
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORICAL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif selected_page == "Historical Analytics":
    section_header(
        "Historical Analytics",
        "Insights from the Cricsheet Men's T20I dataset."
    )

    df = load_data()
    if df is None:
        data_unavailable_warning()
        st.stop()

    # Date range
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    st.caption(f"Dataset period: {date_min} to {date_max} — {len(df)} matches")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Matches by Year", "Team Performance", "Venues", "Toss Analysis", "Head-to-Head"
    ])

    with tab1:
        st.markdown("### Matches per Calendar Year")
        df["year"] = df["date"].dt.year
        year_counts = df.groupby("year").size().reset_index(name="Matches")
        st.bar_chart(year_counts.set_index("year")["Matches"])

        # Show figure if exists
        fig_path = FIGURES_DIR / "matches_by_year.png"
        if fig_path.exists():
            st.image(str(fig_path))
        st.dataframe(year_counts, use_container_width=True)

    with tab2:
        st.markdown("### Team Win Counts and Win Rates")
        # Compute total wins per team
        t1_wins = df[df["target"] == 1]["team1"].value_counts().rename("wins_as_team1")
        t2_wins = df[df["target"] == 0]["team2"].value_counts().rename("wins_as_team2")
        t1_played = df["team1"].value_counts().rename("played_as_team1")
        t2_played = df["team2"].value_counts().rename("played_as_team2")

        teams_all = sorted(set(df["team1"].dropna()) | set(df["team2"].dropna()))
        team_stats = []
        for team in teams_all:
            played = int(t1_played.get(team, 0)) + int(t2_played.get(team, 0))
            wins = int(t1_wins.get(team, 0)) + int(t2_wins.get(team, 0))
            if played >= 5:  # Show only teams with enough matches
                win_rate = wins / played if played > 0 else 0
                team_stats.append({
                    "Team": team,
                    "Matches Played": played,
                    "Wins": wins,
                    "Win Rate": round(win_rate, 3),
                })

        team_df = pd.DataFrame(team_stats).sort_values("Wins", ascending=False)
        st.caption("Showing teams with at least 5 matches. Win rate = wins / total matches played.")
        st.dataframe(team_df.reset_index(drop=True), use_container_width=True)

        fig_path = FIGURES_DIR / "team_win_counts.png"
        if fig_path.exists():
            st.image(str(fig_path))

    with tab3:
        st.markdown("### Venue Distribution")
        venue_counts = df["venue"].value_counts().head(20).reset_index()
        venue_counts.columns = ["Venue", "Matches"]
        st.caption("Top 20 venues by number of matches hosted.")
        st.dataframe(venue_counts, use_container_width=True)

        fig_path = FIGURES_DIR / "venue_distribution.png"
        if fig_path.exists():
            st.image(str(fig_path))

    with tab4:
        st.markdown("### Toss Analysis")
        if "toss_decision" in df.columns:
            toss_dec = df["toss_decision"].value_counts().reset_index()
            toss_dec.columns = ["Decision", "Count"]
            st.markdown("**Toss Decision Distribution**")
            st.dataframe(toss_dec, use_container_width=True)

        # Does toss winner win the match?
        if "toss_winner" in df.columns:
            df_t = df.copy()
            df_t["toss_winner_won"] = (
                ((df_t["toss_winner"] == df_t["team1"]) & (df_t["target"] == 1))
                | ((df_t["toss_winner"] == df_t["team2"]) & (df_t["target"] == 0))
            )
            toss_win_pct = df_t["toss_winner_won"].mean() * 100
            st.metric(
                "Toss Winner Match Win Rate",
                f"{toss_win_pct:.1f}%",
                help="Percentage of matches where the team that won the toss also won the match."
            )
            st.caption(
                "Note: A win rate near 50% suggests the toss has limited predictive value on its own. "
                "Correlation does not imply causation."
            )

        fig_path = FIGURES_DIR / "toss_analysis.png"
        if fig_path.exists():
            st.image(str(fig_path))

    with tab5:
        st.markdown("### Head-to-Head Records")
        st.caption("Select two teams to see their historical T20I record against each other.")

        all_teams = sorted(set(df["team1"].dropna()) | set(df["team2"].dropna()))
        h2h_col1, h2h_col2 = st.columns(2)
        with h2h_col1:
            h2h_team1 = st.selectbox("Team A", all_teams, key="h2h_t1")
        with h2h_col2:
            h2h_team2_opts = [t for t in all_teams if t != h2h_team1]
            h2h_team2 = st.selectbox("Team B", h2h_team2_opts, key="h2h_t2")

        h2h_matches = df[
            ((df["team1"] == h2h_team1) & (df["team2"] == h2h_team2))
            | ((df["team1"] == h2h_team2) & (df["team2"] == h2h_team1))
        ].copy()

        if h2h_matches.empty:
            st.info(f"No T20I matches found between {h2h_team1} and {h2h_team2}.")
        else:
            total_h2h = len(h2h_matches)
            # Wins for each team
            t1_h2h_wins = len(h2h_matches[
                ((h2h_matches["team1"] == h2h_team1) & (h2h_matches["target"] == 1))
                | ((h2h_matches["team2"] == h2h_team1) & (h2h_matches["target"] == 0))
            ])
            t2_h2h_wins = total_h2h - t1_h2h_wins

            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Total Matches", total_h2h)
            hc2.metric(f"{h2h_team1} Wins", t1_h2h_wins)
            hc3.metric(f"{h2h_team2} Wins", t2_h2h_wins)

            h2h_display = h2h_matches[["date", "team1", "team2", "venue", "winner"]].sort_values(
                "date", ascending=False
            )
            st.dataframe(h2h_display.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif selected_page == "Model Performance":
    section_header(
        "Model Performance",
        "Evaluation on the chronological held-out test set (final 20% of matches by date)."
    )

    metrics = load_metrics()
    if metrics is None:
        model_unavailable_warning()
        st.stop()

    # Test period info
    st.markdown("### Test Set Information")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Train Matches", metrics.get("train_rows", "N/A"))
    info_col2.metric("Test Matches", metrics.get("test_rows", "N/A"))
    info_col3.metric("Test Period Start", metrics.get("test_date_min", "N/A"))
    info_col4.metric("Test Period End", metrics.get("test_date_max", "N/A"))

    st.caption(
        "Training used the earliest 80% of matches. "
        "The test set contains the latest 20% — these matches were never seen during training or tuning."
    )
    st.warning(
        "Good historical test performance does not guarantee accurate future predictions. "
        "Match outcomes depend on factors not captured by this model."
    )
    st.markdown("---")

    # Model comparison table
    st.markdown("### Model Comparison Table")
    models_data = metrics.get("models", {})
    best_model_name = metrics.get("best_model", "")

    if models_data:
        rows = []
        for name, m in models_data.items():
            rows.append({
                "Model": name + (" ★" if name == best_model_name else ""),
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1": "{:.4f}", "ROC-AUC": "{:.4f}",
            }).highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                color="#d4edda"
            ),
            use_container_width=True
        )

        st.caption(
            "★ = selected best model. Sorted by ROC-AUC. "
            "Positive class = Team 1 wins (target=1)."
        )

    # Metric explanations
    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
            | Metric | Meaning |
            |--------|---------|
            | **Accuracy** | % of matches correctly predicted |
            | **Precision** | Of all predicted Team1 wins, how many were actually Team1 wins |
            | **Recall** | Of all actual Team1 wins, how many did the model correctly identify |
            | **F1** | Harmonic mean of Precision and Recall |
            | **ROC-AUC** | Area under the ROC curve; measures how well the model ranks Team1 wins above Team2 wins; 0.5 = random, 1.0 = perfect |
            """
        )

    # Charts from outputs/figures/
    st.markdown("### Visualisations")

    fig_tabs = st.tabs(["Model Comparison", "Confusion Matrix", "ROC Curve", "Feature Importance"])

    with fig_tabs[0]:
        fig_path = FIGURES_DIR / "model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path))
        else:
            st.info("Model comparison chart not yet generated. Run training to generate it.")

    with fig_tabs[1]:
        fig_path = FIGURES_DIR / "confusion_matrix.png"
        if fig_path.exists():
            st.image(str(fig_path))
            st.caption(
                "Rows = actual class. Columns = predicted class. "
                "Diagonal = correct predictions."
            )
        else:
            st.info("Confusion matrix not yet generated.")

    with fig_tabs[2]:
        fig_path = FIGURES_DIR / "roc_curve.png"
        if fig_path.exists():
            st.image(str(fig_path))
            st.caption(
                "The closer the curve to the top-left corner, the better. "
                "The dashed line represents a random classifier (AUC = 0.50)."
            )
        else:
            st.info("ROC curve not yet generated.")

    with fig_tabs[3]:
        fig_path = FIGURES_DIR / "feature_importance.png"
        if fig_path.exists():
            st.image(str(fig_path))
            st.caption(
                "Feature importance shows which inputs most influence the model's predictions. "
                "Higher = more influential."
            )
        else:
            st.info("Feature importance chart not available for this model type.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════

elif selected_page == "About Project":
    section_header("About This Project")

    st.markdown(
        """
        ## Problem Statement

        Cricket match outcomes are inherently uncertain. Can historical team performance data,
        toss information, and venue context be used to predict the winner of a Men's T20I match
        more reliably than chance?

        This project builds a supervised binary classification model that, given post-toss
        information, estimates the probability that **Team 1** (the reference team) wins.

        ## Objectives

        1. Acquire real historical T20I match data from Cricsheet.
        2. Engineer meaningful, leak-free features from historical match records.
        3. Train and compare multiple ML algorithms using a chronological split.
        4. Select the best model based on validation evidence.
        5. Deploy an interactive prediction application using Streamlit.

        ---

        ## Dataset Source and Attribution

        - **Provider:** [Cricsheet](https://cricsheet.org) — created by Stephen Rushe
        - **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
        - **Attribution:** Data provided by Cricsheet (https://cricsheet.org)
        - **Format:** Per-match info CSV files from the `t20s_male_csv2.zip` archive

        ---

        ## Feature Engineering

        All historical features are computed using **only matches that occurred before**
        the match being predicted:

        | Feature | Description |
        |---------|-------------|
        | team1 / team2 | Team identities (categorical) |
        | venue | Match venue (categorical) |
        | toss_winner_enc | Toss winner encoded as "team1" or "team2" |
        | toss_decision | bat or field |
        | match_year | Calendar year |
        | team1_win_rate | Team 1 overall historical win rate |
        | team2_win_rate | Team 2 overall historical win rate |
        | team1_recent_form | Team 1 win rate in last 5 matches |
        | team2_recent_form | Team 2 win rate in last 5 matches |
        | h2h_team1_win_rate | Head-to-head win rate for Team 1 vs Team 2 |

        **No post-match information is used.** The winner, winning margin, and final
        score are explicitly excluded from features.

        For teams or pairs with no prior history, a prior of **0.5** (equal probability)
        is used as a documented neutral fallback.

        ---

        ## ML Algorithms

        | Algorithm | How it works |
        |-----------|-------------|
        | **Logistic Regression** | Fits a linear boundary in feature space; outputs a probability via the sigmoid function. Fast, interpretable. |
        | **Decision Tree** | Recursively splits data using feature thresholds. Very interpretable but prone to overfitting. |
        | **Random Forest** | Builds many trees on random data/feature subsets, averages predictions. Robust and less prone to overfitting. |
        | **Gradient Boosting** | Builds trees sequentially; each corrects previous errors. Often high accuracy. |

        ---

        ## Evaluation Approach

        - **Train/Test Split:** Chronological (80% train / 20% test by date)
        - **Cross-Validation:** TimeSeriesSplit (5 folds) — prevents future leakage
        - **Model Selection:** Best ROC-AUC on validation + test evidence
        - **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

        ---

        ## Technologies Used

        | Category | Technologies |
        |----------|-------------|
        | Language | Python 3.11 |
        | Data | pandas, numpy |
        | ML | scikit-learn |
        | Visualisation | matplotlib, seaborn |
        | App | Streamlit |
        | Serialisation | joblib |
        | Testing | pytest |
        | Version Control | Git, GitHub |

        ---

        ## Limitations

        1. **No player-level data** — team selection, player form, and injuries are not included.
        2. **No pitch or weather data** — conditions heavily influence T20I outcomes.
        3. **Historical bias** — teams that have played more matches have more reliable statistics.
        4. **Venue labelling** — the same ground may appear under different names.
        5. **Class balance** — team1/team2 assignment is arbitrary, so the model cannot
           distinguish a "home" advantage in this framing.
        6. **Probability calibration** — the model outputs have not been formally calibrated;
           treat them as relative rankings, not precise probabilities.

        ---

        ## Future Scope

        - Include player-level statistics (batting averages, bowling economy, etc.)
        - Add weather and pitch condition data.
        - Train on ball-by-ball data for in-match live prediction.
        - Implement formal probability calibration (Platt scaling / isotonic regression).
        - Expand to ODI and Test formats.
        - Deploy as a web API for programmatic access.

        ---

        ## References

        - Cricsheet: https://cricsheet.org
        - scikit-learn documentation: https://scikit-learn.org
        - Streamlit documentation: https://docs.streamlit.io
        - pandas documentation: https://pandas.pydata.org
        """
    )
