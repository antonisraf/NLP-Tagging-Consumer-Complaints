import re 
 
 # Function to clean the text 
def clean_transformer_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'X{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text