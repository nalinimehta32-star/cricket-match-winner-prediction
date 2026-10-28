# Cricket Match Winner Prediction — Live Demonstration Guide
## 5 to 7 Minute Step-by-Step Presentation Script

This script provides an engaging, professional walk-through designed for academic reviews, capstone presentations, and viva demonstrations.

---

### Phase-by-Phase Timing Breakdown

| Step | Duration | Target Topic |
| :--- | :---: | :--- |
| **1. Hook & Problem Statement** | 0:00 – 1:00 | Real-world problem, objective, and post-toss timing |
| **2. Dataset & Integrity** | 1:00 – 2:00 | Cricsheet dataset, cleaning, and leak-free feature engineering |
| **3. Exploratory Insights** | 2:00 – 3:00 | Key analytics (Toss myth vs reality, matches by year) |
| **4. ML Models & Benchmarks** | 3:00 – 4:15 | 4 algorithms compared on chronological test set |
| **5. Live Streamlit App Demo** | 4:15 – 6:00 | Interactive match prediction and symmetry verification |
| **6. Summary & Defense** | 6:00 – 7:00 | Architecture, test coverage, and transparent limitations |

---

### Spoken Presentation Script

#### Step 1: Introduction & Problem Statement (0:00 – 1:00)
> *"Good morning respected evaluators. Today I am presenting our Machine Learning project: **Cricket Match Winner Prediction**.*  
> *In Twenty20 International cricket, outcomes are highly unpredictable. Traditional media often rely on anecdotal commentary or isolated head-to-head records. Our goal was to build a rigorous, leak-free, supervised classification system that predicts the probability of victory immediately after the toss but before the first ball is bowled.*  
> *By designating an arbitrary reference Team 1 versus Team 2, our model predicts whether Team 1 wins, outputting continuous win probabilities rather than simplistic binary claims."*

#### Step 2: Real Dataset & Leakage Prevention (1:00 – 2:00)
> *"For our data, we refused to use synthetic data or club leagues like IPL. Instead, we acquired the official Men's T20I archive directly from **Cricsheet** under Creative Commons licensing.*  
> *From 3,646 raw metadata files, we processed 3,527 completed, valid matches spanning February 2005 to October 2026, covering 109 nations and 337 international stadiums.*  
> *Crucially, we avoided future data leakage. naive sports models often calculate win rates over the entire dataset, which leaks future information. In our implementation, every match's historical win rate and 5-match rolling form are calculated strictly using matches that took place prior to that match date."*

#### Step 3: Exploratory Findings & The Toss Myth (2:00 – 3:00)
> *"Our exploratory data analysis revealed several key findings:*  
> *First, we examined whether winning the toss decides the match. In popular folklore, fans assume the toss is everything. However, in our dataset of 3,527 T20Is, the toss winner won the match only **51.8%** of the time—practically a coin flip.*  
> *Second, our target class distribution is naturally balanced at 48.9% to 51.1%, eliminating the need for synthetic oversampling like SMOTE.*  
> *Third, historical win rates and rolling 5-match form showed the strongest positive correlations with match victory."*

#### Step 4: Machine Learning Algorithms & Test Results (3:00 – 4:15)
> *"We trained and benchmarked four algorithms:*  
> 1. *Logistic Regression as an interpretable linear baseline.*  
> 2. *Decision Tree Classifier for rule-based thresholds.*  
> 3. *Random Forest Classifier using bagging over 100 decision trees.*  
> 4. *Gradient Boosting Classifier using sequential error correction.*  
>  
> *Rather than a random train/test split, we enforced a chronological split: the earliest 80% (2,821 matches) for training and hyperparameter tuning with TimeSeriesSplit, and the final 20% (706 matches from late 2025 through late 2026) as our untouched test set.*  
> *On this holdout set, **Gradient Boosting** achieved the best discrimination with an **ROC-AUC of 0.7889** and an **Accuracy of 70.82%**, outperforming Random Forest (0.7877) and Logistic Regression (0.7844)."*

#### Step 5: Live Streamlit Application Demo (4:15 – 6:00)
*(Action: Open the browser running `streamlit run app.py`)*
> *"Let us now inspect the live interactive Streamlit application.*  
> *On the **Home Page**, we see our live metrics: 3,527 matches, 109 nations, and the complete data science pipeline.*  
> *Switching to the **Match Prediction** tab, let us simulate a high-stakes match: **India** as Team 1 versus **Australia** as Team 2 at the **Melbourne Cricket Ground**.*  
> *We select the toss details: India won the toss and elected to bat. Clicking 'Predict Match Winner' instantly computes the pre-match historical features up to today's date.*  
> *Notice that the application clearly outputs estimated probabilities with visual progress bars, accompanied by an explicit educational disclaimer.*  
> *Furthermore, if we reverse the teams—setting Australia as Team 1 and India as Team 2—the probabilities invert symmetrically, proving our model is free of positional bias."*

#### Step 6: Engineering Rigor, Test Suite & Limitations (6:00 – 7:00)
*(Action: Show the terminal with `pytest tests/ -v` passing 40/40)*
> *"To ensure professional software engineering standards, we implemented 40 automated pytest unit tests covering data schemas, target validity, zero-leakage verification, unknown category safety, and app imports. All 40 tests pass cleanly.*  
> *In conclusion, while our model achieves ~71% accuracy on future matches, cricket will always retain inherent unpredictability due to player injuries, pitch changes, and dropped catches.*  
> *Thank you, and I welcome any questions from the panel."*
