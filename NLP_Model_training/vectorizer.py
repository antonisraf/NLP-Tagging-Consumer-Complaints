# model_training.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

# We perform back translation only for the 'Loan Acquisition' group due to its limited sample size (~772 samples).
loan_acquisition_issues = [
    'Getting a loan',
    'Issue where my lender is my school',
    'Issue with income share agreement'
]

# Checking how many samples we have for the 'Loan Acquisition' group before augmentation
loan_acq_df = nlp_data[nlp_data['Issue'].isin(loan_acquisition_issues)].copy()
print(f"Loan Acquisition samples for augmentation: {len(loan_acq_df)}")

# Back-translate the 'Loan Acquisition' samples to augment the dataset
augmented_rows = back_translate_dataframe(
    loan_acq_df,
    text_column='Consumer complaint narrative'
)

# Create a DataFrame from the augmented rows and concatenate with the original data
augmented_df = pd.DataFrame(augmented_rows)
print(f"Successful augmentation: {len(augmented_df)}/{len(loan_acq_df)}")

# Concatenate the augmented data with the original dataset
nlp_data = pd.concat([nlp_data, augmented_df], ignore_index=True)
print(f"Total samples after augmentation: {len(nlp_data)}")

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

# TF-IDF vectorization of the cleaned text data
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

# Fit the TF-IDF vectorizer and transform the cleaned text data into sparse vectors
X_tfidf = tfidf.fit_transform(nlp_data['cleaned_text'])
print(f"\nTF-IDF shape: {X_tfidf.shape}")

# Save the artifacts for later use
save_npz('data/tfidf_vectors.npz', X_tfidf)
joblib.dump(tfidf, 'data/tfidf_vectorizer.pkl')
nlp_data.to_csv('data/student_loan_augmented.csv', index=False)

print("\nSaved:")
print("  data/tfidf_vectors.npz")
print("  data/tfidf_vectorizer.pkl")
print("  data/student_loan_augmented.csv")