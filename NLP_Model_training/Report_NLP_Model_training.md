# Module Overview: Text Preprocessing & TF-IDF Vectorization

This module contains the text preprocessing, data augmentation, and feature extraction pipeline for processing student loan consumer complaints. The pipeline is designed to prepare raw textual narrative data for machine learning classification models using traditional TF-IDF vectorization.

---

## 1. File Descriptions

### `utils_NLP.py`
A utility module containing helper functions for text cleaning, data augmentation, and label mapping.
* **`clean_transformer_text(text)`**: Specialized cleaning for Transformer-based models (retains case, removes URLs and redaction markers like `XXXX`).
* **`clean_tfidf_text(text)`**: Aggressive preprocessing specialized for TF-IDF vectorization. It handles lowercasing, stopword removal, lemmatization, and punctuation/number removal to reduce vocabulary size and noise.
* **`back_translate(text, mid_lang)`**: Leverages the Google Translate API to translate text into an intermediate language (German) and back to English to generate paraphrased variations for data augmentation.
* **`back_translate_dataframe(df, ...)`**: Uses multithreading via `ThreadPoolExecutor` to execute back-translations concurrently across a pandas DataFrame.
* **`get_issue_mapping()`**: Dictates the structural grouping of raw `Issue` labels into 3 broad semantic classes.
* **`get_valid_subissues()`**: Contains a list of highly frequent `Sub-issue` categories used to filter out low-frequency classes.

### `vectorizer.py`
The main execution script that controls the data preparation and vectorization workflow. It loads the dataset, cleans the text fields, applies targeted data augmentation, splits the data, fits the TF-IDF vectorizer, and saves the final processed data splits and model artifacts.

---

## 2. Pipeline Logic & Methodology

### Text Preprocessing Logic
The pipeline distinguishes between deep learning text preparation and statistical vocabulary extraction. For TF-IDF, the script uses `clean_tfidf_text` to normalize tokens:
* **Noise Reduction**: Characters matching numbers, punctuation, and specific credit/loan content masking tokens (e.g., `XXXX`, `XX`) are stripped away.
* **Dimensionality Minimization**: Lemmatization via NLTK's `WordNetLemmatizer` converts verbs to their base form (e.g., *running*, *runs* $\rightarrow$ *run*), ensuring that morphological variations do not inflate the feature space dimensions.

### Label Engineering & Grouping
To combat extreme class sparsity and improve classification performance, data grouping logic is applied to target labels:
* **Issue Mapping**: Original complaints span numerous distinct issues. These are compressed into three semantic categories: `Loan Management`, `Credit Report Issues`, and `Loan Acquisition`. Infrequent categories with fewer than 200 samples are dropped entirely.
* **Sub-Issue Filtering**: Sub-issues containing fewer than 500 total occurrences are dynamically re-assigned to a catch-all `'Other'` class to prevent high-variance errors in downstream models.

### Data Split & Augmentation Strategy
1.  **Stratified Splitting**: The dataset is split into an $80/20$ train/test ratio using stratification on the grouped `Issue` label. This guarantees balanced class representations across both validation and training sets.
2.  **Targeted Back-Translation**: To correct data imbalance, data augmentation via back-translation is isolated strictly to the minority class (`Loan Acquisition`) within the training split. **Augmentation is never applied to the test set** to ensure evaluation metrics remain un-tainted by synthetic data.
3.  **ASCII Filtering**: Paraphrased strings returned from the translation engine containing non-ASCII properties are discarded to preserve uniform vocabulary tokenization.

### TF-IDF Configuration
The `TfidfVectorizer` transforms the normalized unstructured text into sparse matrices using the following hyperparameters:
* `max_features=50000`: Limits vocabulary to the top 50,000 most informative terms.
* `ngram_range=(1, 2)`: Captures both individual words and two-word phrases (unigrams and bigrams) to preserve local context (e.g., "bad information").
* `min_df=3`: Ignores rare terms appearing in fewer than 3 separate documents to eliminate unique typos.
* `max_df=0.95`: Discards corpus-wide terms appearing in more than 95% of documents (words with low predictive value).
* `sublinear_tf=True`: Scales term frequency logarithmically ($1 + \log(\text{tf})$), reducing the influence of repetitive keywords within unusually long narratives.

---

## 3. Data Flow Architecture

The data transitions through the following file paths and structures during execution:

## Student Loan NLP Pipeline

```
data/student_loan_nlp_clean.csv
            │
            ▼
  Text Cleaning & Label Mapping
            │
            ▼
  Stratified Train/Test Split (80/20)
            │
    ┌───────┴───────┐
    ▼               ▼
[Train Split]   [Test Split]
    │               │
    ▼               │
Targeted            │
Back-Translation    │
(Loan Acq. Only)    │
    │               │
    ▼               ▼
Fit & Transform  Transform
   TF-IDF         Only
    └───────┬───────┘
            │
            ▼
      Saved Artifacts:
  ├── data/X_train_tfidf.npz
  ├── data/X_test_tfidf.npz
  ├── data/tfidf_vectorizer.pkl
  ├── data/student_loan_augmented.csv
  └── data/student_loan_test.csv
```