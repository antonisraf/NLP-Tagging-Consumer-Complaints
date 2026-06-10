# Module Overview: Text Preprocessing & TF-IDF Vectorization

This module contains the text preprocessing, data augmentation, and feature extraction pipeline for processing student loan consumer complaints. The pipeline is designed to prepare raw textual narrative data for machine learning classification models using traditional TF-IDF vectorization.

---

## 1. File Descriptions

### `utils_NLP.py`
A utility module containing helper functions for text cleaning, data augmentation, label mapping, and vocabulary-based filtering.

* **`clean_tfidf_text(text)`**: Aggressive preprocessing specialized for TF-IDF vectorization. Handles lowercasing, URL removal, redaction marker removal (`XXXX`), number and punctuation stripping, stopword removal, single-character token filtering, and lemmatization using a noun-first/verb-fallback strategy to reduce vocabulary size and noise.
* **`back_translate(text, mid_lang)`**: Leverages the Google Translate API to translate text into an intermediate language (German) and back to English to generate paraphrased variations for data augmentation. Texts exceeding `BACK_TRANSLATE_MAX_CHARS` (5000 characters) are skipped entirely to avoid augmenting incomplete or truncated samples.
* **`back_translate_dataframe(df, ...)`**: Uses multithreading via `ThreadPoolExecutor` to execute back-translations concurrently across a pandas DataFrame. Rate limiting is enforced via a `threading.Semaphore` to ensure requests are spaced out across threads rather than firing simultaneously.
* **`get_issue_mapping()`**: Dictates the structural grouping of raw `Issue` labels into 3 broad semantic classes.
* **`filter_by_vocab_count(df, text_column, min_unique, max_unique)`**: Filters rows based on the number of unique tokens in the cleaned text. Removes entries with too few unique tokens (likely empty or junk text) and entries with too many unique tokens (likely data dumps or malformed entries). Applied before train/test split to avoid leakage. Returns the filtered DataFrame and a stats dictionary.
* **`get_valid_subissues()`**: Returns the list of `Sub-issue` labels retained after frequency filtering. Sub-issues with more than 500 samples are kept as distinct classes; all remaining Sub-issues are grouped under `'Other'`.

### `vectorizer.py`
The main execution script that controls the data preparation and vectorization workflow. It loads the dataset, cleans the text fields, applies vocabulary-based filtering, applies targeted data augmentation, splits the data, fits the TF-IDF vectorizer, and saves the final processed data splits and model artifacts.

---

## 2. Pipeline Logic & Methodology

### Text Preprocessing Logic
For TF-IDF, the script uses `clean_tfidf_text` to normalize tokens:
* **Noise Reduction**: Characters matching numbers, punctuation, and specific credit/loan content masking tokens (e.g., `XXXX`) are stripped away. Single-character tokens are also removed as they carry no semantic value.
* **Dimensionality Minimization**: Lemmatization via NLTK's `WordNetLemmatizer` uses a noun-first/verb-fallback strategy — each token is first lemmatized as a noun, and if the form is unchanged, it is re-lemmatized as a verb. This ensures morphological variations do not inflate the feature space.

### Vocabulary-Based Filtering
After text cleaning and before the train/test split, rows are filtered based on unique token count:
* Rows with **fewer than 5 unique tokens** are removed as they are likely empty or junk entries after preprocessing.
* Rows with **more than 500 unique tokens** are removed as they are likely malformed data dumps.
* Filtering is applied before the split to prevent data leakage. Stats (rows removed, percentage kept, mean/median token counts) are printed for transparency.

### Label Engineering & Grouping
To combat extreme class sparsity and improve classification performance, data grouping logic is applied to target labels:
* **Issue Mapping**: Original complaints span numerous distinct issues. These are compressed into three semantic categories: `Loan Management`, `Credit Report Issues`, and `Loan Acquisition`. Infrequent categories with fewer than 200 samples are dropped entirely.
* **Sub-Issue Filtering**: Sub-issues with fewer than 500 total occurrences are dynamically re-assigned to a catch-all `'Other'` class to prevent high-variance errors in downstream models.

### Data Split & Augmentation Strategy
1. **Stratified Splitting**: The dataset is split into an 80/20 train/test ratio using stratification on the grouped `Issue` label. This guarantees balanced class representations across both sets.
2. **Targeted Back-Translation**: To correct data imbalance, data augmentation via back-translation is isolated strictly to the minority class (`Loan Acquisition`) within the training split. **Augmentation is never applied to the test set** to ensure evaluation metrics remain untainted by synthetic data.
3. **Character Limit Skipping**: Texts exceeding 5000 characters are skipped entirely during back-translation (rather than truncated) to avoid generating augmented samples from incomplete narratives.
4. **ASCII Filtering**: Paraphrased strings containing non-ASCII characters are discarded to preserve uniform vocabulary tokenization.

### TF-IDF Configuration
The `TfidfVectorizer` transforms the normalized text into sparse matrices using the following hyperparameters:
* `max_features=50000`: Limits vocabulary to the top 50,000 most informative terms.
* `ngram_range=(1, 2)`: Captures both individual words and two-word phrases (unigrams and bigrams) to preserve local context (e.g., "bad information").
* `min_df=3`: Ignores rare terms appearing in fewer than 3 documents to eliminate unique typos.
* `max_df=0.95`: Discards terms appearing in more than 95% of documents (words with low predictive value).
* `sublinear_tf=True`: Scales term frequency logarithmically (1 + log(tf)), reducing the influence of repetitive keywords within long narratives.

---

## 3. Data Flow Architecture

The data transitions through the following file paths and structures during execution:

## Student Loan NLP Pipeline

```
data/student_loan_nlp_clean.csv
            │
            ▼
  Text Cleaning (clean_tfidf_text)
            │
            ▼
  Vocabulary-Based Filtering
  (filter_by_vocab_count)
            │
            ▼
  Label Mapping & Sub-issue Grouping
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
