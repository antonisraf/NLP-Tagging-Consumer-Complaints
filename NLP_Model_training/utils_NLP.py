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
        noun_form = lemmatizer.lemmatize(t, pos='n')
        lemma = noun_form if noun_form != t else lemmatizer.lemmatize(t, pos='v')
        cleaned.append(lemma)
    return ' '.join(cleaned)


def back_translate(text, mid_lang='de'):
    """
    Applies back-translation for data augmentation.
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
    Returns an optimized mapping dictionary that aligns Level 1 Issues 
    1-to-1 with Level 2 semantic groups to ensure structural harmony.
    """
    return {
        # 1. Loan Management -> Αντιστοιχεί στο Loan Information & Servicing
        'Dealing with your lender or servicer':                      'Loan Information & Servicing',
        
        # 2. Repayment Issues -> Αντιστοιχεί στο Payment & Repayment Issues
        'Struggling to repay your loan':                            'Payment & Repayment Issues',
        
        # 3. Credit Report Issues -> Αντιστοιχεί στο Credit Reporting Issues
        'Incorrect information on your report':                     'Credit Reporting Issues',
        'Improper use of your report':                              'Credit Reporting Issues',
        "Problem with a company's investigation into an existing problem": 'Credit Reporting Issues',
        'Unable to get your credit report or credit score':         'Credit Reporting Issues',
        "Problem with a credit reporting company's investigation into an existing problem": 'Credit Reporting Issues',
        
        # 4. Loan Acquisition -> Αντιστοιχεί στο Loan Acquisition & Eligibility
        'Getting a loan':                                           'Loan Acquisition & Eligibility',
        'Issue where my lender is my school':                       'Loan Acquisition & Eligibility',
        'Issue with income share agreement':                        'Loan Acquisition & Eligibility',
    }


def filter_by_vocab_count(df, text_column, min_unique=5, max_unique=500):
    """
    Filters DataFrame rows based on the number of unique tokens in the cleaned text.
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

def get_subissue_mapping():
    """
    Returns the mapping dictionary for grouping Sub-issue labels into 4 semantic groups.
    """
    payment     = 'Payment & Repayment Issues'
    servicing   = 'Loan Information & Servicing'
    credit      = 'Credit Reporting Issues'
    acquisition = 'Loan Acquisition & Eligibility'

    return {
        # Payment & Repayment Issues
        'Trouble with how payments are being handled':              payment,
        "Don't agree with the fees charged":                       payment,
        'Problem with your payment plan':                          payment,
        "Can't get other flexible options for repaying your loan": payment,
        "Can't temporarily delay making payments":                 payment,
        'Problem lowering your monthly payments':                  payment,
        'Issues with fees connected to the loan':                  payment,
        'Payment issues':                                          payment,
        'Billing or statement issues':                             payment,
        'Billing dispute for services':                            payment,

        # Loan Information & Servicing
        'Received bad information about your loan':                servicing,
        'Problem with customer service':                           servicing,
        'Need information about your loan balance or loan terms':  servicing,
        'Problem with forgiveness, cancellation, or discharge':    servicing,
        'Changes in terms mid-deal or after closing':              servicing,
        'Problem with the interest rate':                          servicing,
        'Marketing or disclosure issues':                          servicing,
        'Confusing or misleading advertising':                     servicing,
        'High pressure sales tactics or recruiting':               servicing,
        "Didn't receive services that were advertised":            servicing,
        'Issues with financial aid services':                      servicing,
        'Problem with product or service terms changing':          servicing,
        'Problem with signing the paperwork':                      servicing,
        'Dealing with provider of income share agreement':         servicing,

        # Credit Reporting Issues
        'Account information incorrect':                           credit,
        'Account status incorrect':                                credit,
        'Reporting company used your report improperly':           credit,
        'Their investigation did not fix an error on your report': credit,
        'Information belongs to someone else':                     credit,
        'Old information reappears or never goes away':            credit,
        'Personal information incorrect':                          credit,
        "Credit inquiries on your report that you don't recognize": credit,
        'Was not notified of investigation status or results':     credit,
        'Investigation took more than 30 days':                    credit,
        'Information is missing that should be on the report':     credit,
        'Public record information inaccurate':                    credit,
        'Other problem getting your report or credit score':       credit,
        'Problem getting your free annual credit report':          credit,
        'Problem canceling credit monitoring or identify theft protection service': credit,
        'Report provided to employer without your written authorization': credit,
        'Problem with personal statement of dispute':              credit,
        'Difficulty submitting a dispute or getting information about a dispute over the phone': credit,

        # Loan Acquisition & Eligibility
        'Co-signer':                                               acquisition,
        'Denied loan':                                             acquisition,
        'Loan opened without my consent or knowledge':             acquisition,
        'Fraudulent loan':                                         acquisition,
        'Qualified for a better loan than the one offered':        acquisition,
        'Bankruptcy':                                              acquisition,
        'Cannot graduate, receive diploma, or get transcript due to money owed': acquisition,
        'Keep getting calls about your loan':                      acquisition,
        'Received unwanted marketing or advertising':              acquisition,
        'Received unsolicited financial product or insurance offers after opting out': acquisition,
    }
