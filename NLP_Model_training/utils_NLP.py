import re
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_transformer_text(text):
    """
    Cleans raw complaint text for use with SentenceTransformer models.

    Steps:
        - Converts input to string
        - Removes URLs
        - Removes redacted placeholders (e.g. XXXX)
        - Collapses multiple whitespace into a single space

    Args:
        text (str): Raw complaint narrative text.

    Returns:
        str: Cleaned text string.
    """
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'X{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_tfidf_text(text):
    """
    Cleans and normalizes raw complaint text specifically for TF-IDF vectorization.

    Applies more aggressive preprocessing than clean_transformer_text, including
    stopword removal, lemmatization, and punctuation removal. These steps reduce
    noise and vocabulary size, improving TF-IDF representation quality.

    Steps:
        - Converts input to string and lowercases
        - Removes URLs
        - Removes redacted placeholders (e.g. XXXX)
        - Removes numbers
        - Removes punctuation
        - Tokenizes and removes English stopwords
        - Applies WordNet lemmatization
        - Collapses multiple whitespace into a single space

    Args:
        text (str): Raw complaint narrative text.

    Returns:
        str: Cleaned and lemmatized text string.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'X{2,}', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)


def back_translate(text, mid_lang='de'):
    """
    Applies back-translation for data augmentation.

    Translates text from English to an intermediate language and back to English.
    Used to generate paraphrased versions of underrepresented complaint narratives.

    Args:
        text (str): Original English complaint text.
        mid_lang (str): Intermediate language code (default: 'de' for German).
                        German is preferred over Greek due to higher translation reliability.

    Returns:
        str or None: Back-translated English text, or None if translation failed
                     or if non-ASCII characters are detected in the result.
    """
    try:
        text = str(text)[:5000]
        translated = GoogleTranslator(source='en', target=mid_lang).translate(text)
        back = GoogleTranslator(source=mid_lang, target='en').translate(translated)
        if any(ord(c) > 127 for c in back):
            return None
        return back
    except Exception:
        return None


def back_translate_dataframe(df, text_column, max_workers=5, sleep=0.05):
    """
    Applies back-translation to all rows of a DataFrame using multithreading.

    Uses ThreadPoolExecutor for parallel API calls to speed up augmentation.
    A small sleep is added between requests to avoid hitting Google Translate rate limits.

    Args:
        df (pd.DataFrame): DataFrame containing the text column to augment.
        text_column (str): Name of the column with complaint narratives.
        max_workers (int): Number of parallel threads (default: 5).
        sleep (float): Sleep time in seconds between requests (default: 0.05).

    Returns:
        list: List of augmented row dicts (only successful translations included).
    """
    def augment_row(row):
        time.sleep(sleep)
        result = back_translate(row[text_column])
        if result:
            new_row = row.copy()
            new_row[text_column] = result
            return new_row
        return None

    rows = [row for _, row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(augment_row, rows),
            total=len(rows),
            desc="Back-translating"
        ))

    return [r for r in results if r is not None]


def get_issue_mapping():
    """
    Returns the mapping dictionary for grouping original Issue labels into 3 semantic groups.

    Groups:
        - Loan Management: Issues related to lender/servicer interactions and repayment.
        - Credit Report Issues: Issues related to credit reporting errors and investigations.
        - Loan Acquisition: Issues related to obtaining or co-signing loans.

    Note:
        Issues not present in this mapping (e.g. Fraud & Security) are excluded
        from the dataset during preprocessing due to insufficient samples (<200).

    Returns:
        dict: Mapping from original Issue label to grouped label.
    """
    return {
        'Dealing with your lender or servicer': 'Loan Management',
        'Struggling to repay your loan': 'Loan Management',
        'Incorrect information on your report': 'Credit Report Issues',
        'Improper use of your report': 'Credit Report Issues',
        "Problem with a company's investigation into an existing problem": 'Credit Report Issues',
        'Unable to get your credit report or credit score': 'Credit Report Issues',
        "Problem with a credit reporting company's investigation into an existing problem": 'Credit Report Issues',
        'Getting a loan': 'Loan Acquisition',
        'Issue where my lender is my school': 'Loan Acquisition',
        'Issue with income share agreement': 'Loan Acquisition',
    }


def get_valid_subissues():
    """
    Returns the list of Sub-issue labels retained after frequency filtering.

    Only Sub-issues with more than 500 samples are kept as distinct classes.
    All remaining Sub-issues are grouped under 'Other'.

    Returns:
        list: List of valid Sub-issue label strings.
    """
    return [
        'Trouble with how payments are being handled',
        'Received bad information about your loan',
        'Problem with customer service',
        'Problem with forgiveness, cancellation, or discharge',
        "Don't agree with the fees charged",
        'Need information about your loan balance or loan terms',
        'Problem with your payment plan',
        'Reporting company used your report improperly',
        "Can't get other flexible options for repaying your loan",
        'Account information incorrect',
        'Account status incorrect',
    ]