import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from utils_NLP import clean_transformer_text

# Load the dataset
nlp_data = pd.read_csv('data/student_loan_nlp_clean.csv')

# Clean the text data using the clean_transformer_text function from utils_NLP.py
nlp_data['cleaned_text'] = nlp_data['Consumer complaint narrative'].apply(clean_transformer_text)

# Load the pre-trained SentenceTransformer model miniLM-L6-v2
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for the cleaned text data
embeddings = model.encode(
    nlp_data['cleaned_text'].tolist(),
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True
)

# Save the embeddings to a .npy file
np.save('data/embeddings.npy', embeddings)
print("Embeddings saved successfully to 'data/embeddings.npy'")
