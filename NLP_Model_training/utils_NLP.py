import re
import time
import threading
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

# Max character limit for back-translation — texts exceeding this are skipped
# to avoid augmenting incomplete/truncated samples
BACK_TRANSLATE_MAX_CHARS = 5000


def clean_tfidf_text(text):
    """
    Cleans and normalizes raw complaint text specifically for TF-IDF vectorization.

    Applies aggressive preprocessing including stopword removal, lemmatization,
    and punctuation removal to reduce noise and vocabulary size.

    Steps:
        - Converts input to string and lowercases
        - Removes URLs
        - Removes redacted placeholders (e.g. XXXX)
        - Removes numbers
        - Removes punctuation
        - Tokenizes and removes English stopwords
        - Removes single-character tokens
        - Applies WordNet lemmatization with noun-first, verb-fallback strategy
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
    cleaned = []
    for t in tokens:
        if len(t) <= 1 or t in stop_words:
            continue
        # Lemmatize as noun first; if unchanged, try as verb
        noun_form = lemmatizer.lemmatize(t, pos='n')
        lemma = noun_form if noun_form != t else lemmatizer.lemmatize(t, pos='v')
        cleaned.append(lemma)
    return ' '.join(cleaned)


def back_translate(text, mid_lang='de'):
    """
    Applies back-translation for data augmentation.

    Translates text from English to an intermediate language and back to English.
    Used to generate paraphrased versions of underrepresented complaint narratives.

    Texts exceeding BACK_TRANSLATE_MAX_CHARS are skipped entirely to avoid
    augmenting incomplete samples caused by truncation.

    Args:
        text (str): Original English complaint text.
        mid_lang (str): Intermediate language code (default: 'de' for German).
                        German is preferred due to higher translation reliability.

    Returns:
        str or None: Back-translated English text, or None if:
                     - text exceeds character limit
                     - translation failed
                     - non-ASCII characters detected in result
    """
    try:
        text = str(text)
        if len(text) > BACK_TRANSLATE_MAX_CHARS:
            return None
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
    Rate limiting is enforced via a threading Semaphore, ensuring requests are
    spaced out across all threads rather than firing simultaneously.

    Args:
        df (pd.DataFrame): DataFrame containing the text column to augment.
        text_column (str): Name of the column with complaint narratives.
        max_workers (int): Number of parallel threads (default: 5).
        sleep (float): Minimum sleep time in seconds between requests (default: 0.05).

    Returns:
        list: List of augmented row dicts (only successful translations included).
    """
    semaphore = threading.Semaphore(1)

    def augment_row(row):
        with semaphore:
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


def filter_by_vocab_count(df, text_column, min_unique=5, max_unique=500):
    """
    Filters DataFrame rows based on the number of unique tokens in the cleaned text.

    Removes rows with too few unique tokens (likely empty or junk text) and rows
    with too many unique tokens (likely data dumps or malformed entries). Applied
    before train/test split to avoid leakage.

    Args:
        df (pd.DataFrame): DataFrame containing the cleaned text column.
        text_column (str): Name of the column with cleaned complaint text.
        min_unique (int): Minimum number of unique tokens required (default: 5).
        max_unique (int): Maximum number of unique tokens allowed (default: 500).

    Returns:
        tuple:
            - pd.DataFrame: Filtered DataFrame (reset index).
            - dict: Stats with keys:
                'original'      : total rows before filtering
                'removed_low'   : rows removed for too few unique tokens
                'removed_high'  : rows removed for too many unique tokens
                'final'         : rows remaining after filtering
                'mean_unique'   : mean unique token count in filtered set
                'median_unique' : median unique token count in filtered set
                'mean_tokens'   : mean total token count in filtered set
                'pct_kept'      : percentage of rows retained
    """
    original = len(df)
    unique_counts = df[text_column].apply(lambda x: len(set(str(x).split())))
    total_counts = df[text_column].apply(lambda x: len(str(x).split()))

    mask_low = unique_counts < min_unique
    mask_high = unique_counts > max_unique
    keep_mask = ~mask_low & ~mask_high

    filtered_df = df[keep_mask].reset_index(drop=True)
    filtered_unique = unique_counts[keep_mask]
    filtered_total = total_counts[keep_mask]

    stats = {
        'original':       original,
        'removed_low':    int(mask_low.sum()),
        'removed_high':   int(mask_high.sum()),
        'final':          len(filtered_df),
        'mean_unique':    round(filtered_unique.mean(), 1),
        'median_unique':  round(filtered_unique.median(), 1),
        'mean_tokens':    round(filtered_total.mean(), 1),
        'pct_kept':       round(len(filtered_df) / original * 100, 2),
    }
    return filtered_df, stats


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