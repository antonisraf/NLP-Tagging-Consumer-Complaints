# ComplaintFlow `app/` Folder README

> **AI-powered complaint classification, human-backed.**

This directory contains the full application layer for **ComplaintFlow**, a Streamlit-based dashboard that samples real, held-out CFPB student loan complaints (2021-2022), classifies them using a pre-trained hierarchical NLP pipeline, and routes low-confidence or incorrect predictions to a human review queue.

---

## Pipeline Overview

![ComplaintFlow Pipeline](complaintflow_pipeline.png)

---

## Use Case Diagram

![ComplaintFlow Use Case Diagram](complaintflow_use_case_diagram.png)

---

## Directory Structure

```
app/
├── streamlit_app.py          # Main Streamlit entry point
├── model_pipeline.py         # Inference wrapper (HierarchicalComplaintClassifier)
├── human_review_section.py   # Human Review Queue UI & logic
├── real_complaint_loader.py  # Samples real held-out CFPB complaints
├── train_and_save_models.py  # One-time training script (run before first launch)
└── logo.png                  # Brand badge displayed in the app header
```

---

## File Reference

### `streamlit_app.py`
The main application entry point. Handles the full user-facing experience:

- **No API key required.** The app no longer calls any external LLM; it reads directly from a local CSV of real, held-out complaints.
- **Three-Tab Layout**:
  - **Activity Log** — chat-style log showing the status of each pipeline run. Controls for the number of complaints (10–30, default 20) and the joint confidence threshold (default `0.45`) are also here.
  - **Results & Analysis** — renders a dashboard with classification accuracy metrics, confusion matrices, and a per-complaint results table. After human review is finalised, the dashboard reflects the updated (human-corrected) labels.
  - **Human Review Queue** — delegates to `human_review_section.py`.
- **Sidebar** — shows whether the held-out data file was found on disk, session controls, CSV download, and a dashboard reset button.
- **Theming** — dark-mode-only glass-morphism UI using CSS custom properties (Inter font, Slate/Indigo palette, radial gradient background).

**Key pipeline flow on "Load & Classify":**
1. Calls `load_real_complaints()` to sample `n` real complaints (with genuine CFPB-assigned labels) from `data/cfpb_2021-2022_holdout.csv`.
2. Runs them through `HierarchicalComplaintClassifier.predict()`.
3. Compares predictions against the real ground-truth labels.
4. Forces `needs_review = True` for any row where the broad issue (Level 1) was predicted incorrectly (`apply_eval_review_override`), **and** additionally for any row where the sub-issue (Level 2) was predicted incorrectly — this second override is applied directly in `streamlit_app.py`, on top of the model's own confidence-based flagging.
5. Stores the full results DataFrame in `st.session_state`.

---

### `model_pipeline.py`
Inference-only wrapper around the hierarchical two-level classifier. Loads two pre-trained artifacts from `data/`:

| Artifact | Description |
|---|---|
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (built by `vectorizer.py`) |
| `hierarchical_model_bundle.pkl` | Full model bundle (built by `train_and_save_models.py`) |

**`HierarchicalComplaintClassifier.predict(texts, threshold)`**

Runs the full cascade:
1. Cleans each text with `clean_tfidf_text()` (imported from `NLP_Model_training/utils_NLP.py`).
2. **Level 1 (broad issue):** Averages probabilities from a `LogisticRegression` and a calibrated `LinearSVC` to predict one of two broad groups: `Loan Servicing & Payments` or `Non-Servicing Issues`.
3. **Level 2 (sub-issue):** For each broad group, runs a separate pair of (LR + SVC) models whose feature matrix includes the Level-1 probability vector appended to the TF-IDF features (feature cascading). Predicts one of four sub-issues.
4. **Joint confidence** = Level-1 confidence × Level-2 confidence. Rows with joint confidence below `threshold` (default `0.45`) are flagged `needs_review = True`.

Returns a `pd.DataFrame` with columns: `complaint_text`, `cleaned_text`, `predicted_issue_broad`, `issue_confidence`, `predicted_subissue`, `subissue_confidence`, `joint_confidence`, `needs_review`, `joint_perplexity`.

**Joint Perplexity**

In addition to `joint_confidence`, `predict()` computes a Joint Perplexity score in the same inference pass, no extra model calls required. It measures the combined uncertainty of the pipeline across both levels, computed directly from the average probability distributions:

Joint Perplexity = e^(H(L1) + H(L2))

Where the entropy H for a probability distribution P is:

H(P) = −Σᵢ Pᵢ · log(Pᵢ + ε)

with ε = 10⁻¹⁰ added for numerical stability, to prevent log(0) errors.

Interpretation:

| Value | Meaning |
|---|---|
| ≈ 1.0 | Absolute certainty. The model assigned ~100% probability to a single broad issue and a single sub-issue. Text is clear and unambiguous. |
| ≈ 2.0 | Low/medium uncertainty. The model's confusion is equivalent to choosing between 2 equally likely scenarios. |
| ≥ 3.0 | High uncertainty. The model is split across 3 or more overlapping outcomes, typically complex, multi-topic, or poorly written complaints that likely require Human Review. |

Exposed in the Streamlit dashboard so reviewers can filter and prioritise the hardest cases, separately from the `needs_review` threshold flag.

**Known limitation — blind spot on near-duplicate categories**

Validated on the real labelled test set, joint perplexity separates correct from incorrect predictions with an overall AUC-ROC of **0.648** (point-biserial r = 0.228). Breaking this down by error type shows the signal is not uniform:

| Error type | n | Mean perplexity | AUC vs. correct |
|---|---|---|---|
| Correct predictions | 2492 | 2.164 | — |
| `Loan Information & Servicing` ↔ `Payment & Repayment Issues` confusion | 911 | 2.292 | **0.601** |
| All other errors | 417 | 2.741 | **0.751** |

`Loan Information & Servicing` and `Payment & Repayment Issues` are near-duplicate categories that share vocabulary. This single pair accounts for roughly two-thirds of all sub-issue errors, and it is exactly where joint perplexity performs weakest (AUC 0.601, barely above chance). The model tends to be **confidently wrong** here rather than genuinely uncertain: the predicted probability distribution is sharp, just pointed at the wrong class, so entropy stays low even though the prediction is wrong.

By contrast, on rarer or more unusual errors outside this pair, perplexity works well (AUC 0.751) and is a reasonable uncertainty signal.

**Implication:** joint perplexity should not be used as the sole or primary routing signal for human review. It is a useful secondary diagnostic for surfacing atypical or multi-topic complaints, but it should not be trusted to catch the dominant Info/Payment confusion. In the app, this specific pair is instead caught by the explicit Level-2 correctness override described in `streamlit_app.py` above, since the Streamlit run has access to real ground-truth labels; a production deployment without ground truth would not have this safety net and would need a dedicated strategy for this pair.

---

### `human_review_section.py`
Renders the **Human Review Queue** tab and manages the review lifecycle.

**Taxonomy enforced:**

| Broad Issue | Sub-Issues |
|---|---|
| Loan Servicing & Payments | Loan Information & Servicing · Payment & Repayment Issues |
| Non-Servicing Issues | Credit Reporting Issues · Loan Acquisition & Eligibility |

**Review flow:**
- Displays only the rows where `needs_review = True`.
- Each complaint card shows the model's prediction, joint confidence, and a status badge (Pending / Decided).
- A reviewer picks a sub-issue from a flat dropdown of all 4 options; the broad issue is derived automatically via `SUBISSUE_TO_ISSUE` reverse lookup.
- Progress bar tracks how many complaints have been decided.
- **Finalise Reviews** is only enabled once all flagged complaints have a decision. Clicking it calls `apply_review_decisions()`, which writes `reviewed_issue`, `reviewed_subissue`, and `review_source` (`"human"` or `"model"`) back to the DataFrame.

**Accuracy treatment after finalisation:**
- Human-reviewed rows are marked `issue_correct = True` and `subissue_correct = True` — the human takes ownership of those labels; they are not scored against the ground truth.
- Model-only rows retain their original accuracy verdicts.

After finalisation the Results tab reflects the updated labels and recomputes metrics accordingly.

---

### `real_complaint_loader.py`
Loads real, held-out CFPB student loan complaints from `data/cfpb_2021-2022_holdout.csv` and returns them in the shape the app expects: a list of dicts with keys `complaint_text`, `true_issue`, `true_subissue`.

**`load_real_complaints(csv_path, n, random_state)`**

- Expects the **raw CFPB export CSV** format (as downloaded from consumerfinance.gov), i.e. the original `Consumer complaint narrative`, `Issue`, `Sub-issue` columns — not a pre-processed file.
- Drops rows with missing/empty narratives.
- Maps `Issue` → `Issue_grouped` and `Sub-issue` → `Subissue_grouped` using the same mappings used at training time (`get_issue_mapping()`, `get_subissue_mapping()`); rows that fall outside the trained taxonomy are dropped.
- Derives `true_issue_broad` via the same `GROUPING` dict used everywhere else in the pipeline (must stay in sync with `train_and_save_models.py`).
- Raises `FileNotFoundError` / `ValueError` with an actionable message if the file is missing or malformed, rather than failing silently — `streamlit_app.py` surfaces these directly in the Activity Log.

Genuine consumer narratives with genuine CFPB-assigned ground-truth labels; no LLM bias/style artifacts, no API key, no generation cost.

---

### `train_and_save_models.py`
**One-time training script.** Run this once before launching the app to produce `data/hierarchical_model_bundle.pkl`.

**What it does:**
1. Loads pre-computed TF-IDF features (`X_train_tfidf.npz`) and the augmented training CSV.
2. Derives Level-1 labels from `Subissue_grouped` using the `GROUPING` dict.
3. Fits a `LogisticRegression` and a calibrated `LinearSVC` for Level 1 (broad issue).
4. Computes **out-of-fold Level-1 probabilities** via 5-fold cross-validation, which are then appended to the TF-IDF features as cascade inputs to Level-2 models (prevents leakage).
5. Fits separate LR + calibrated SVC models for each of the two broad groups at Level 2.
6. Serialises all models, class arrays, and the grouping map into a single `joblib` bundle.

**Usage:**
```bash
python app/train_and_save_models.py
```

**Output:** `data/hierarchical_model_bundle.pkl`

> The `GROUPING` dict here must remain in sync with the taxonomy in `real_complaint_loader.py` and `human_review_section.py`.

---

## Prerequisites & Setup

### Data files required (in `data/`)
| File | Built by |
|---|---|
| `student_loan_augmented.csv` | Provided / data pipeline |
| `tfidf_vectorizer.pkl` | `vectorizer.py` |
| `X_train_tfidf.npz` | `vectorizer.py` |
| `hierarchical_model_bundle.pkl` | `train_and_save_models.py` |
| `cfpb_2021-2022_holdout.csv` | Raw CFPB export, filtered to Product = Student loan, years 2021-2022 (outside the training window) — download manually and place here |

### First-time setup
```bash
# 1. Build the TF-IDF vectorizer (if not already done)
python NLP_Model_training/vectorizer.py

# 2. Train and save the hierarchical model bundle
python app/train_and_save_models.py

# 3. Place the CFPB 2021-2022 holdout export at data/cfpb_2021-2022_holdout.csv

# 4. Launch the app
streamlit run app/streamlit_app.py
```

---

## Notes

- **No external API calls at runtime.** The previous version of this app generated synthetic complaints via the Groq API on every run; that path has been fully removed. All evaluation data now comes from real, held-out CFPB complaints (2021-2022), which sit outside the 2023–early 2026 training window used to build the model.
- All classification logic is **inference-only** at runtime; no retraining happens when the app is running.
- Because the held-out complaints carry genuine CFPB-assigned labels rather than LLM-generated ones, accuracy figures in the Results tab reflect real-world performance rather than the model's ability to match another model's (the generator's) labelling conventions.
- Human review decisions override model predictions and are propagated to the Results tab as soon as the reviewer clicks **Finalise Reviews**.
