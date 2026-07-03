
"""
Loads real, held-out CFPB student loan complaints (e.g. year 2022, outside
the 2023-early 2026 training window) to use as evaluation data instead of
LLM-generated synthetic complaints.

Replaces complaint_generator.py: no API key, no LLM bias/style artifacts,
genuine consumer narratives with genuine CFPB-assigned ground-truth labels.

Expects the RAW CFPB export CSV (as downloaded from
https://www.consumerfinance.gov/data-research/consumer-complaints/), i.e.
the same column names as the original training data before any of the
project's own preprocessing: 'Consumer complaint narrative', 'Issue',
'Sub-issue'. Any other columns in the export (Date received, Product,
Company, etc.) are ignored.
"""
import os
import sys
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _BASE_DIR not in sys.path:
    sys.path.append(_BASE_DIR)

from NLP_Model_training.utils_NLP import get_issue_mapping, get_subissue_mapping

# Must stay in sync with the GROUPING dict in train_and_save_models.py /
# model_evaluation.py, this is what turns the 4-class Subissue_grouped
# label into the 2-class broad Level-1 target.
GROUPING = {
    "Credit Reporting Issues": "Non-Servicing Issues",
    "Loan Acquisition & Eligibility": "Non-Servicing Issues",
    "Loan Information & Servicing": "Loan Servicing & Payments",
    "Payment & Repayment Issues": "Loan Servicing & Payments",
}


def _prepare_holdout_df(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Held-out complaints file not found at {csv_path}. "
            "Download the raw CFPB export (Product = Student loan, filtered "
            "to the desired year) and place it at this path."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"Consumer complaint narrative", "Issue", "Sub-issue"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Held-out CSV is missing expected column(s): {missing}. "
            "Use the raw CFPB export format, not a pre-processed file."
        )

    df = df.dropna(subset=["Consumer complaint narrative"]).reset_index(drop=True)
    df = df[df["Consumer complaint narrative"].str.strip() != ""].reset_index(drop=True)

    # Same Issue -> Issue_grouped mapping used at training time. Rows whose
    # Issue isn't in the mapping fall outside the taxonomy the model was
    # trained on and are dropped, exactly as vectorizer.py does.
    issue_mapping = get_issue_mapping()
    df["Issue_grouped"] = df["Issue"].map(issue_mapping)
    df = df.dropna(subset=["Issue_grouped"]).reset_index(drop=True)

    # Same Sub-issue -> Subissue_grouped mapping. Unmapped sub-issues become
    # 'Other' and are dropped since the trained classifier has no 'Other'
    # class to predict.
    subissue_mapping = get_subissue_mapping()
    df["Sub-issue"] = df["Sub-issue"].fillna("Other")
    df["Subissue_grouped"] = df["Sub-issue"].map(subissue_mapping).fillna("Other")
    df = df[df["Subissue_grouped"] != "Other"].reset_index(drop=True)

    df["true_issue_broad"] = df["Subissue_grouped"].map(GROUPING)

    return df


def load_real_complaints(csv_path, n=10, random_state=None):
    """
    Samples `n` real, held-out complaints from `csv_path` (raw CFPB export
    format) and returns them in the same shape previously returned by
    generate_synthetic_complaints(): a list of dicts with keys
    'complaint_text', 'true_issue', 'true_subissue'.

    Raises FileNotFoundError / ValueError with an actionable message if the
    file is missing or malformed, rather than failing silently.
    """
    df = _prepare_holdout_df(csv_path)

    if len(df) == 0:
        raise ValueError(
            "No usable rows remained after filtering to the trained "
            "taxonomy. Check that the CSV actually contains student loan "
            "complaints with narratives."
        )

    n = min(n, len(df))
    sample = df.sample(n=n, random_state=random_state)

    records = []
    for _, row in sample.iterrows():
        records.append({
            "complaint_text": str(row["Consumer complaint narrative"]).strip(),
            "true_issue":     row["true_issue_broad"],
            "true_subissue":  row["Subissue_grouped"],
        })
    return records