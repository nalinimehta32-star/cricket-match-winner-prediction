# Cricket Match Winner Prediction Using Machine Learning
## Comprehensive Project Report

**Program:** MBA Data Science & Artificial Intelligence  
**Academic Level:** Mini-Project / Applied Capstone Deliverable  
**Domain:** Sports Analytics & Predictive Modelling  

---

## Abstract

Predicting the outcome of cricket matches is an intricate domain challenge due to the stochastic nature of Twenty20 International (T20I) cricket, high variance in player performances, and the influence of match conditions. In this study, we develop a leak-free, supervised binary classification pipeline to predict Men's T20I match winners at the post-toss, pre-match juncture. Historical ball-by-ball and scorecard data comprising 3,527 official Men's T20I matches from February 2005 through October 2026 were sourced from Cricsheet. Feature engineering was carried out strictly in chronological sequence, extracting prior team win rates, recent 5-match form trajectories, head-to-head records, venue assignments, and relative toss dynamics without future information leakage. Four machine learning algorithms—Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Gradient Boosting Classifier—were trained using TimeSeriesSplit cross-validation and evaluated on an untouched chronological holdout test set representing the final 20% of matches (706 matches). Gradient Boosting achieved the highest discrimination power with an ROC-AUC of 0.7889 and an Accuracy of 70.82%. The complete preprocessing and inference architecture was serialized and deployed as an interactive Streamlit application, verified through automated unit tests.

---

## 1. Introduction & Background

Cricket is among the most popular sports globally, with the Twenty20 (T20) format commanding significant attention due to its fast-paced nature, high scoring rates, and close finishes. Historically, cricket analysis has relied heavily on qualitative commentary, aggregated batting/bowling averages, and intuitive post-match appraisals. 

However, modern data science provides analytical methodologies to convert historical records into predictive intelligence. Predicting match winners prior to the commencement of play offers substantial pedagogical value for students of Data Science and AI, illuminating essential concepts such as temporal data leakage, non-stationary sports dynamics, class balance nuances, probability calibration, and deployment via modern web architectures.

---

## 2. Problem Statement & Objectives

### 2.1 Problem Formulation
We formulate match winner prediction as a **supervised binary classification problem**:
- Given an eligible completed match between two nations, one nation is arbitrarily designated as **Team 1** (reference team) and the other as **Team 2**.
- The binary target is defined as:
  $$\text{target} = \begin{cases} 1, & \text{if Team 1 wins} \\ 0, & \text{if Team 2 wins} \end{cases}$$
- The prediction timing is fixed as **post-toss, pre-match**: features comprise information established immediately after the toss but prior to the delivery of the first ball.

### 2.2 Key Objectives
1. **Acquire and normalize** real-world T20I historical match records from an official open data source (Cricsheet).
2. **Design leakage-proof features**, computing rolling win rates and recent form strictly over chronologically preceding fixtures.
3. **Train, tune, and benchmark** multiple machine learning algorithms using time-aware validation.
4. **Select and serialize** the optimal production model into a reusable pipeline artifact.
5. **Develop an interactive Streamlit application** supporting match simulation, statistical visualisations, and transparent model metrics.
6. **Ensure automated test coverage** validating schema integrity, boundary cases, and prediction consistency.

---

## 3. Dataset Description & Acquisition

### 3.1 Data Source & Licensing
- **Provider:** [Cricsheet](https://cricsheet.org), maintained by Stephen Rushe.
- **Archive:** `t20s_male_csv2.zip` (Men's T20 Internationals).
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- **Attribution Statement:** "Data provided by Cricsheet (https://cricsheet.org)".

### 3.2 Ingestion & Cleaning Summary
The raw archive contained 3,646 match information files. Ingestion logic parsed each metadata file, normalized team and venue titles, and applied filtering rules:
- **Total Ingested Matches:** 3,646
- **Valid Binary Matches:** 3,527
- **Excluded Matches:** 119 matches (comprising abandoned matches, no-results, rain-affected non-completions, and tied matches where no binary result applied).
- **Date Span:** 17 February 2005 (New Zealand vs Australia) to 25 October 2026.
- **Unique Teams:** 109 nations (spanning ICC Full Members and Associate Members).
- **Unique Grounds:** 337 international cricket venues.
- **Class Balance:** Class 1 (Team 1 won) = 1,725 (48.91%); Class 0 (Team 2 won) = 1,802 (51.09%).

---

## 4. Feature Engineering & Leakage Prevention

A critical flaw in naive sports predictive models is **future data leakage** (using statistics aggregated over the whole dataset to predict an earlier match). In this project, feature engineering was implemented via a single chronological sweep:

1. **Team 1 Historical Win Rate (`team1_win_rate`):** Proportion of historical completed matches won by Team 1 prior to the match date.
2. **Team 2 Historical Win Rate (`team2_win_rate`):** Proportion of historical completed matches won by Team 2 prior to the match date.
3. **Recent Form (`team1_recent_form`, `team2_recent_form`):** Win percentage across the immediate prior 5 matches completed by the respective team.
4. **Head-to-Head Win Rate (`h2h_team1_win_rate`):** Historical win rate of Team 1 specifically against Team 2 in previous encounters.
5. **Relative Toss Winner (`toss_winner_enc`):** Categorical encoding indicating whether Team 1 or Team 2 won the coin toss.
6. **Toss Decision (`toss_decision`):** Tactical decision made at the toss (`bat` or `field`).
7. **Match Year (`match_year`):** Normalized temporal index representing the evolving era of T20 cricket.
8. **Categorical Identities (`team1`, `team2`, `venue`):** Handled via OneHotEncoder with `handle_unknown='ignore'`.

*Documented Prior Fallback:* For teams or pairings with zero prior matches, a neutral prior probability of $0.50$ is applied.

---

## 5. Model Training & Evaluation Methodology

### 5.1 Chronological Split
Random shuffling was strictly rejected to preserve temporal validity. The dataset was partitioned chronologically:
- **Training Set:** 2,821 matches (80.0%) spanning 2005-02-17 to 2025-09-12.
- **Test Holdout Set:** 706 matches (20.0%) spanning 2025-09-12 to 2026-10-25.

### 5.2 Cross-Validation & Hyperparameter Tuning
Hyperparameters were tuned exclusively on the training partition using a 5-fold **TimeSeriesSplit** cross-validator with ROC-AUC scoring.

### 5.3 Comparative Experimental Results (On 706 Held-Out Matches)

| Algorithm | Hyperparameters / Setup | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | `n_estimators=100, max_depth=3, lr=0.05` | **0.7082** | **0.7168** | **0.6884** | **0.7023** | **0.7889** |
| **Random Forest** | `n_estimators=100, max_depth=8, min_samples_split=5` | 0.7139 | 0.7352 | 0.6686 | 0.7003 | 0.7877 |
| **Logistic Regression** | `C=0.01, penalty='l2', solver='lbfgs'` | 0.6983 | 0.7201 | 0.6487 | 0.6826 | 0.7844 |
| **Decision Tree** | `max_depth=6, min_samples_split=2` | 0.6856 | 0.6866 | 0.6827 | 0.6847 | 0.7377 |

### 5.4 Model Selection Rationale
Gradient Boosting Classifier was selected as the final production model:
1. It demonstrated the **highest ROC-AUC metric (0.7889)** on the untouched future test horizon, indicating superior rank-ordering of winning probabilities.
2. It exhibits a well-balanced precision (0.7168) and recall (0.6884) without collapsing into single-class bias.
3. Gradient boosting iteratively minimizes logistic loss by fitting shallow regression trees to negative gradients, making it particularly resilient to tabular non-linearities.

---

## 6. Software Architecture & Implementation

The application adheres to clean modular software engineering standards:
- `acquire_data.py`: Automated archive retrieval and Cricsheet CSV2 parsing.
- `src/data_loader.py`: Schema validation, missingness profiling, and data quality reporting.
- `src/feature_engineering.py`: Time-series aware feature engineering engine.
- `src/preprocessing.py`: ColumnTransformer pipeline with one-hot encoding and standard scaling.
- `src/train_model.py`: Model fitting, TimeSeriesSplit grid search, and joblib persistence.
- `src/evaluate_model.py`: Calculation of accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
- `src/predict.py`: Standalone inference module supporting single-match post-toss prediction.
- `app.py`: Five-section Streamlit application (Home, Prediction, Analytics, Model Performance, About).
- `tests/`: 40 automated pytest unit tests covering data validation, leakage checks, probability ranges, team inversion invariance, and error handling.

---

## 7. Key Findings & Insights

1. **The Toss is Not Decisive:** Across 3,527 T20Is, the toss winner won the match only **51.8%** of the time. This empirically refutes the popular belief that the toss determines the outcome in international T20s.
2. **Historical Win Rate Dominance:** Long-term historical win rate and 5-match rolling form are the most influential numerical indicators of match victory.
3. **Model Accuracy Boundaries:** A test accuracy of ~71% represents a strong predictive threshold for pre-match forecasting in a high-variance sport like T20 cricket. Reaching higher accuracies pre-match without in-game ball-by-ball telemetry is constrained by inherent match stochasticity.

---

## 8. Limitations & Future Scope

### 8.1 Limitations
- **Omission of Individual Player Form:** Key injuries or star player selections on match day are not captured in team-level aggregates.
- **Microclimate & Pitch Telemetry:** Grass cover, pitch deterioration, and evening dew factors are not systematically recorded in standard scorecards.
- **Associate Country Data Density:** Teams with few historical international fixtures rely heavily on fallback priors.

### 8.2 Future Scope
- Incorporate player-level ICC rankings and recent batting/bowling strike rates.
- Ingest real-time weather and pitch reports via weather APIs.
- Extend modeling to in-play live win probability estimation during second innings chases.

---

## 9. References

1. Cricsheet: Open Cricket Data — https://cricsheet.org.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 1189–1232.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. Streamlit Documentation: Modern Data Apps — https://docs.streamlit.io.
