# Cricket Match Winner Prediction — Comprehensive Viva Preparation Guide
## 32 Master Questions and Natural Answers for Viva / Oral Examination

This guide prepares students for academic viva voce examinations, project reviews, and technical presentations. Answers are concise, technically grounded, and directly reflect the code implemented in this repository.

---

### Q1: What is Machine Learning?
**Answer:** Machine Learning is a subfield of artificial intelligence where algorithms learn patterns directly from historical data to make predictions or decisions, rather than relying on strictly hardcoded deterministic rules.

### Q2: What is Supervised Learning, and how does it apply here?
**Answer:** Supervised learning is an ML paradigm where models are trained on labelled data containing both input features ($X$) and known true outcomes ($y$). In our project, the input features are pre-match parameters (teams, venue, toss, historical form), and the ground truth label is whether Team 1 won the match.

### Q3: Why is this project formulated as a classification problem rather than regression?
**Answer:** The outcome of a completed cricket match is a discrete, categorical decision: either Team 1 wins or Team 2 wins (a binary event). Because the target is discrete rather than a continuous numerical quantity (such as total runs scored), binary classification is the mathematically sound formulation.

### Q4: What is the exact target variable used in this project?
**Answer:** The target variable is `target`, a binary integer:
- `target = 1`: The reference team (`team1`) won the match.
- `target = 0`: The opposing team (`team2`) won the match.

### Q5: Why predict whether "Team 1" wins rather than predicting a specific country like India?
**Answer:** In cricket, matches take place between arbitrary pairs of 100+ different countries. By formulating the target generically as "Did reference Team 1 defeat Team 2?", a single universal model can predict any bilateral match between any two teams in the world without requiring separate models for each nation.

### Q6: What is Data Leakage, and why is it dangerous in sports analytics?
**Answer:** Data leakage occurs when information from outside the training partition or from the future is inadvertently introduced into model training. In cricket, if we calculate a team's win rate using matches that happened *after* the match being predicted, or if we include final match scores, margins of victory, or Player of the Match awards as features, the model will exhibit artificially inflated performance in testing but fail completely in real-world deployment.

### Q7: Why did you include historical win rate as a feature?
**Answer:** A team's historical track record reflects its long-term baseline strength, cricketing infrastructure, and overall calibre. High historical win rates over a significant sample of matches correlate with competitive capability.

### Q8: How did you calculate "recent form" in your feature engineering?
**Answer:** Recent form is computed as the win percentage of a team across their last 5 completed matches immediately preceding the current fixture date. It captures short-term momentum, team rhythm, and confidence.

### Q9: Why did you use a chronological train/test split instead of a standard random split?
**Answer:** Cricket matches occur in a strict time series. A random shuffle split would leak future match outcomes into the past (e.g., training on 2026 matches to predict a 2012 match). A chronological split (earliest 80% for training: 2005 to 2025; latest 20% for testing: 2025 to 2026) mirrors realistic forward-looking prediction.

### Q10: How does Logistic Regression work?
**Answer:** Logistic Regression is a linear classification algorithm. It calculates a linear combination of input features weighted by coefficients, and then applies the sigmoid (logistic) function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
This transforms the output into a smooth probability value between 0 and 1, where values $\ge 0.5$ indicate a predicted Team 1 victory.

### Q11: How does a Decision Tree work?
**Answer:** A Decision Tree recursively splits data based on feature value thresholds (e.g., `team1_win_rate > 0.58`) to maximize purity (minimizing Gini impurity or entropy). It creates an interpretable hierarchy of if-else rules ending at terminal leaf nodes that assign a class label.

### Q12: How does Random Forest work, and why is it generally better than a single Decision Tree?
**Answer:** Random Forest is an ensemble technique (bagging). It trains hundreds of individual decision trees, each on a random bootstrap sample of the dataset and a random subset of features. Predictions from all individual trees are averaged (or majority voted). This substantially reduces variance and guards against the overfitting prone to single decision trees.

### Q13: How does Gradient Boosting work?
**Answer:** Gradient Boosting builds trees sequentially rather than independently. Each new shallow tree is explicitly trained to predict and correct the residual errors (pseudo-residuals) of the previous ensemble of trees, minimizing a differentiable loss function (log-loss).

### Q14: What is overfitting, and how did you prevent it?
**Answer:** Overfitting occurs when a model learns noise, idiosyncrasies, and random fluctuations in the training data rather than generalisable underlying patterns, resulting in high training performance but poor test performance. We prevented overfitting by:
1. Using TimeSeriesSplit cross-validation.
2. Limiting tree depth (`max_depth=3` in Gradient Boosting, `max_depth=8` in Random Forest).
3. Using regularisation (`l2` penalty in Logistic Regression).
4. Evaluating on an untouched chronological holdout set.

### Q15: What is the difference between Accuracy, Precision, and Recall in this project?
**Answer:**
- **Accuracy:** The overall percentage of matches correctly predicted ($\frac{TP+TN}{Total}$).
- **Precision:** Of all matches where the model predicted Team 1 would win, what fraction did Team 1 actually win ($\frac{TP}{TP+FP}$)?
- **Recall:** Of all matches where Team 1 actually won, what fraction did the model successfully identify ($\frac{TP}{TP+FN}$)?

### Q16: What is the F1-Score, and why is it useful?
**Answer:** F1-score is the harmonic mean of precision and recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It provides a single balanced metric, ensuring that a model does not simply maximize precision at the catastrophic expense of recall, or vice versa.

### Q17: What is ROC-AUC, and what was your best model's ROC-AUC?
**Answer:** ROC-AUC stands for Area Under the Receiver Operating Characteristic curve. It plots True Positive Rate against False Positive Rate across all possible classification probability thresholds. An AUC of 0.5 denotes random guessing, while 1.0 denotes perfect separation. Our best model (Gradient Boosting) achieved an **ROC-AUC of 0.7889** on the chronological test set.

### Q18: What is a Confusion Matrix?
**Answer:** A Confusion Matrix is a $2 \times 2$ table showing the count of True Positives, True Negatives, False Positives, and False Negatives produced by a classifier on the test dataset, illustrating specific types of prediction errors.

### Q19: Why is cross-validation important, and why did you use TimeSeriesSplit?
**Answer:** Cross-validation assesses how well a model generalizes across different data subsets during training. Standard k-fold cross-validation randomly shuffles observations, which would cause temporal leakage in time-series data. `TimeSeriesSplit` creates rolling chronological folds where the training set always precedes the validation set in time.

### Q20: What is Hyperparameter Tuning?
**Answer:** Hyperparameters are configuration variables set before training (such as tree depth, learning rate, or regularisation parameter $C$) that cannot be learned directly from data. Tuning systematically tests parameter combinations using cross-validation to select the values that maximize performance.

### Q21: Why is toss information included in the model?
**Answer:** The coin toss gives the winning captain tactical choice of whether to bat first or bowl first, allowing them to leverage ground pitch behavior, morning moisture, or evening dew. In our dataset, it provides a realistic post-toss pre-match feature.

### Q22: Why is this explicitly defined as a "post-toss, pre-match" predictor?
**Answer:** Because toss outcome and toss decision are unknown until approximately 30 minutes before match commencement. Describing the model as a "pre-toss" predictor while feeding it toss variables would be factually incorrect.

### Q23: What is Feature Engineering?
**Answer:** Feature engineering is the process of transforming raw domain data into structured explanatory numerical and categorical variables that facilitate effective pattern recognition by machine learning algorithms.

### Q24: How did you guarantee that no future-information leakage occurred during feature engineering?
**Answer:** In `src/feature_engineering.py`, we iterate through matches sorted by ascending date. For each match $i$, we compute rolling statistics strictly from matches $0$ to $i-1$. The outcome of match $i$ is only added to the historical accumulation records *after* its own feature row has been computed.

### Q25: How was the final model selected?
**Answer:** The final model was selected by comparing all four algorithms on the 706-match chronological holdout set. Gradient Boosting achieved the highest ROC-AUC (0.7889), outperforming Random Forest (0.7877), Logistic Regression (0.7844), and Decision Tree (0.7377), demonstrating the strongest discriminative capability.

### Q26: How does the Streamlit web application interact with the trained model?
**Answer:** The Streamlit app loads the persisted pipeline artifact (`models/best_model.joblib`) and feature metadata (`models/feature_metadata.json`). When a user enters match parameters, the app computes historical features dynamically up to the selected match date and passes them to the fitted pipeline's `.predict()` and `.predict_proba()` methods.

### Q27: What is Joblib, and why is it used instead of Python's default Pickle?
**Answer:** `joblib` is an optimized serialization library designed for NumPy arrays and scikit-learn estimators. It handles large numerical matrices efficiently and serializes complete scikit-learn Pipelines including both preprocessing transformers and estimators.

### Q28: What is Git, and what is GitHub?
**Answer:** Git is a distributed version control system running locally on a developer's machine to track source code changes over time. GitHub is a cloud-based hosting platform for Git repositories that enables collaboration, remote backups, and code sharing.

### Q29: What are the primary limitations of your model?
**Answer:**
1. It does not ingest player-level statistics (e.g., individual bowler economy or batter strike rate).
2. It lacks environmental telemetry (weather, overcast conditions, pitch deterioration, and evening dew).
3. Infrequently playing associate nations have limited historical sample sizes, falling back onto a neutral 0.50 prior.

### Q30: Why can the model be wrong in real life?
**Answer:** Cricket is inherently stochastic. Individual performances (e.g., a batsman playing a career-best innings or a dropped catch) can alter match outcomes independently of historical team statistics. Additionally, match-day injuries, tactical fielding shifts, and weather interventions are outside the model's feature space.

### Q31: How did you handle team order reversal symmetry?
**Answer:** We wrote automated tests verifying that swapping Team 1 and Team 2 correctly swaps the predicted probabilities so that $P(\text{Team 1 wins}) + P(\text{Team 2 wins}) \approx 1.0$, preventing any positional bias toward whoever is selected as Team 1.

### Q32: How could this project be improved in future work?
**Answer:** Future extensions could ingest real-time ball-by-ball commentary data for live in-play win probability estimation (like Duckworth-Lewis-Stern and WASP engines), incorporate player-level ICC rankings, integrate automated weather APIs, and apply formal Platt scaling or Isotonic Regression for calibrated probabilities.
