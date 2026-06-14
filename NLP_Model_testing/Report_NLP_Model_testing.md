# Module Overview: Hierarchical Model Training & Evaluation

This module contains the model training and evaluation scripts for the two-level (Issue → Sub-issue) hierarchical classifier built on top of the TF-IDF artifacts produced by the vectorization module. Both scripts consume the same saved artifacts (`X_train_tfidf.npz`, `X_test_tfidf.npz`, `student_loan_augmented.csv`, `student_loan_test.csv`, `tfidf_vectorizer.pkl`) but represent two different stages of the experiment.

---

## 1. File Descriptions

### `hierarchical_baseline_experiment.py`
An exploratory experiment script that runs the full hierarchical pipeline with hyperparameter search and produces a set of diagnostic reports. It trains a Level 1 (Issue) classifier on the 4-class grouping defined by `get_issue_mapping()`, cascades the Level 1 probabilities into per-group Level 2 (Sub-issue) classifiers, and then separately tests an alternative 2-class Level 1 grouping for comparison. It also runs a threshold-based rejection analysis and prints full classification reports for every stage.

### `model_evaluation.py`
The operational evaluation script that builds a fixed, simplified version of the hierarchical pipeline using a 2-class Level 1 target derived directly from `Subissue_grouped`. It uses fixed hyperparameters (no grid search), introduces a joint Level 1 × Level 2 confidence score, sweeps the rejection threshold to study the automation/review trade-off, and generates a 3-panel visual dashboard summarizing the results.

---

## 2. Pipeline Logic & Methodology

### Two-Level Hierarchical Structure
Both scripts follow the same general shape: a Level 1 model predicts a broad group, its predicted probabilities are fed as additional features (via `hstack` with the TF-IDF matrix) into a separate Level 2 model trained only on the rows belonging to that group. Out-of-fold (OOF) Level 1 probabilities are computed via `cross_val_predict` for the training set so that the Level 2 models never see Level 1 probabilities that were produced by a model trained on the same rows.

### Level 1: Broad Issue Classification
* **`hierarchical_baseline_experiment.py`**: Level 1 target is `Issue_grouped`, a 4-class label set (`Loan Information & Servicing`, `Payment & Repayment Issues`, `Credit Reporting Issues`, `Loan Acquisition & Eligibility`) coming from `get_issue_mapping()`. Both Logistic Regression and a calibrated `LinearSVC` are tuned via `GridSearchCV` over `C in [0.01, 0.1, 0.5, 1.0]`, using 5-fold stratified CV and `f1_macro` scoring.
* **`model_evaluation.py`**: Level 1 target is a 2-class label (`Loan Servicing & Payments` vs `Non-Servicing Issues`) derived directly from `Subissue_grouped` via a local `GROUPING` dict, not from `Issue_grouped`. Both models use a fixed `C=1.0` with no grid search.

### Soft-Vote Ensemble & Out-of-Fold Cascading
In both scripts, Level 1 predictions are produced by averaging the predicted probabilities of the Logistic Regression model and the calibrated `LinearSVC` ("soft voting"). The same averaging is applied to the OOF probabilities used as Level 2 input features.

### Level 2: Sub-issue Classification with Feature Cascading
For each Level 1 group, a separate Level 2 model is trained on `hstack([TF-IDF features, Level 1 probability features])`, with the target being `Subissue_grouped`.
* **`hierarchical_baseline_experiment.py`**: Uses `GridSearchCV` over `C in [0.1, 1.0, 5.0]` for both LR and the calibrated `LinearSVC`. Includes edge-case handling: groups with only one Sub-issue class are assigned that class directly with confidence 1.0 and no model is trained; groups where the smallest class has fewer than 2 samples skip the SVC entirely and use LR only.
* **`model_evaluation.py`**: Uses fixed `C=1.0` for both LR and `LinearSVC` for every group, with no grid search and no special-casing for single-class or very small groups.

### Confidence Scoring & Rejection Threshold
* **`hierarchical_baseline_experiment.py`**: The routing confidence is the Level 2 model's own maximum predicted probability (`avg_sub_proba.max(axis=1)`). A single fixed `REJECTION_THRESHOLD = 0.45` is used to split predictions into "auto-labelled" and "sent for review."
* **`model_evaluation.py`**: The routing confidence is a joint score, `level1_confidence * max(P(Level 2))`, combining how confident the model is about the broad group with how confident it is about the specific sub-issue within that group. This score is swept across thresholds from 0.30 to 0.85 in steps of 0.05 to study the trade-off between the auto-labelled subset's Macro F1 and the human-review rate, and a chosen threshold of 0.45 is used for the final report.

### Diagnostics & Reporting
* **`hierarchical_baseline_experiment.py`**:
  * Confusion matrix and top confusion-pair table for the Level 1 (4-class) predictions.
  * A second, independent experiment that maps `Issue_grouped` to a 2-class alternative grouping (`GROUPING_ALTERNATIVE`) and trains a fresh Level 1 model (with its own `GridSearchCV`) on that target for comparison.
  * Full `classification_report` output for Level 1, Level 2 (all predictions), and Level 2 (auto-labelled subset only).
* **`model_evaluation.py`**:
  * Confusion matrices for Level 1 (2-class) and for Level 2 on the auto-labelled subset at the chosen threshold.
  * A 3-panel dashboard saved to `plots/nlp_performance_dashboard.png`:
    1. Threshold trade-off curve (Auto Subset Macro F1 vs. Human Review %), with the chosen threshold (0.45) marked.
    2. Histogram/KDE of the joint confidence score `P(L1) × P(L2)` across all test predictions, with the threshold marked.
    3. Donut chart showing the auto-labelled vs. human-review split at the chosen threshold.

---

## 3. Inputs & Outputs

### Inputs (both scripts)
```
data/X_train_tfidf.npz
data/X_test_tfidf.npz
data/student_loan_augmented.csv
data/student_loan_test.csv
data/tfidf_vectorizer.pkl
```

### Outputs
* `hierarchical_baseline_experiment.py`: console output only (classification reports, confusion matrices, confidence distribution, summary table). No files are saved.
* `model_evaluation.py`: console output (classification reports and confusion matrices at the chosen threshold) plus `plots/nlp_performance_dashboard.png`.

---


