import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
import joblib
from utils_NLP import (
    clean_tfidf_text,
    back_translate_dataframe,
    get_issue_mapping,
    get_valid_subissues
)

# Load the cleaned dataset
nlp_data = pd.read_csv('data/student_loan_nlp_clean.csv')

# Clean the text using TF-IDF specific preprocessing (lemmatization, stopword removal, etc.)
nlp_data['cleaned_text'] = nlp_data['Consumer complaint narrative'].apply(clean_tfidf_text)

# Group the original Issues into 3 semantic groups
issue_mapping = get_issue_mapping()
nlp_data['Issue_grouped'] = nlp_data['Issue'].map(issue_mapping)
nlp_data = nlp_data.dropna(subset=['Issue_grouped'])
nlp_data = nlp_data.reset_index(drop=True)

print("\n=== Issue distribution ===")
print(nlp_data['Issue_grouped'].value_counts())

# Keep only Sub-issues with more than 500 samples, grouping the rest under 'Other'
valid_subissues = get_valid_subissues()
nlp_data['Sub-issue'] = nlp_data['Sub-issue'].fillna('Other')
nlp_data['Subissue_grouped'] = nlp_data['Sub-issue'].apply(
    lambda x: x if x in valid_subissues else 'Other'
)

print("\n=== Sub-issue distribution ===")
print(nlp_data['Subissue_grouped'].value_counts())

# Split into train/test before augmentation and TF-IDF fitting
train_df, test_df = train_test_split(
    nlp_data,
    test_size=0.2,
    random_state=42,
    stratify=nlp_data['Issue_grouped']
)

print(f"\nTrain size: {len(train_df)} | Test size: {len(test_df)}")

# We perform back translation only for the 'Loan Acquisition' group due to its limited sample size (~772 samples)
# Augmentation is applied only to the training set
loan_acquisition_issues = [
    'Getting a loan',
    'Issue where my lender is my school',
    'Issue with income share agreement'
]

# Checking how many samples we have for the 'Loan Acquisition' group before augmentation
loan_acq_train = train_df[train_df['Issue'].isin(loan_acquisition_issues)].copy()
print(f"\nLoan Acquisition samples for augmentation: {len(loan_acq_train)}")

# Back-translate the 'Loan Acquisition' samples to augment the dataset
augmented_rows = back_translate_dataframe(
    loan_acq_train,
    text_column='Consumer complaint narrative'
)

# Create a DataFrame from the augmented rows and concatenate with the original training data
augmented_df = pd.DataFrame(augmented_rows)
augmented_df['cleaned_text'] = augmented_df['Consumer complaint narrative'].apply(clean_tfidf_text)
print(f"Successful augmentation: {len(augmented_df)}/{len(loan_acq_train)}")

# Concatenate the augmented data with the training dataset only
train_df = pd.concat([train_df, augmented_df], ignore_index=True)
print(f"Total train samples after augmentation: {len(train_df)}")

# TF-IDF vectorization: fit only on train set, transform both train and test
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

# Fit the TF-IDF vectorizer on training data only, then transform both sets
X_train = tfidf.fit_transform(train_df['cleaned_text'])
X_test = tfidf.transform(test_df['cleaned_text'])
print(f"\nTF-IDF shape — Train: {X_train.shape} | Test: {X_test.shape}")

# Save the artifacts for later use
save_npz('data/X_train_tfidf.npz', X_train)
save_npz('data/X_test_tfidf.npz', X_test)
joblib.dump(tfidf, 'data/tfidf_vectorizer.pkl')
train_df.to_csv('data/student_loan_augmented.csv', index=False)
test_df.to_csv('data/student_loan_test.csv', index=False)
