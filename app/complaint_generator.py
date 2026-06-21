"""
Generates synthetic student-loan complaint narratives via the Groq API.
Returns dicts with keys: complaint_text, true_issue, true_subissue
"""
import json
import os
import pandas as pd
from groq import Groq

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DATA_DIR  = os.path.join(BASE_DIR, "data")

MODEL = "llama-3.3-70b-versatile"

ISSUE_SUBISSUE_MAP = {
    "Loan Servicing & Payments": [
        "Loan Information & Servicing",
        "Payment & Repayment Issues",
    ],
    "Non-Servicing Issues": [
        "Credit Reporting Issues",
        "Loan Acquisition & Eligibility",
    ],
}


def _sample_examples(n=4):
    train_path = os.path.join(DATA_DIR, "student_loan_augmented.csv")
    try:
        train_df = pd.read_csv(train_path)
        pool = train_df["Consumer complaint narrative"].dropna()
        if pool.empty:
            return []
        return pool.sample(n=min(n, len(pool))).tolist()
    except FileNotFoundError:
        print(f"WARNING: Training data not found at {train_path}. Skipping few-shot examples.")
        return []
    except Exception as e:
        print(f"WARNING: Could not load examples: {e}")
        return []


def _extract_json_array(raw_text):
    try:
        start_idx = raw_text.find('[')
        end_idx   = raw_text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return json.loads(raw_text[start_idx:end_idx + 1])
        print("No JSON brackets found in response.")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []


def generate_synthetic_complaints(n=10, api_key=None, topics=None, num_examples=4):
    client = Groq(api_key=api_key) if api_key else Groq()

    BATCH_SIZE = 10
    all_cleaned_records = []

    batches = [BATCH_SIZE] * (n // BATCH_SIZE)
    if n % BATCH_SIZE != 0:
        batches.append(n % BATCH_SIZE)

    for batch_n in batches:
        examples = _sample_examples(num_examples)

        if examples:
            examples_block = "\n\n".join(
                f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples)
            )
            examples_section = (
                "Below are real anonymized complaint narratives shown ONLY as "
                "style/tone/length references (do NOT copy their wording):\n\n"
                + examples_block + "\n"
            )
        else:
            examples_section = ""

        topic_hint = ""
        if topics:
            topic_hint = (
                "\nTry to cover a mix of these topics across the complaints "
                "(one or more complaints per topic): " + "; ".join(topics) + "."
            )

        prompt = f"""You are generating SYNTHETIC test data for an NLP classifier that
categorizes CFPB student loan complaints.

The classifier uses EXACTLY this two-level taxonomy — use no other strings:

  Broad issue: "Loan Servicing & Payments"
    Sub-issue A: "Loan Information & Servicing"
      → complaints about servicer errors, wrong loan info, poor customer service.
    Sub-issue B: "Payment & Repayment Issues"
      → complaints about payment processing errors, income-driven repayment, auto-pay.

  Broad issue: "Non-Servicing Issues"
    Sub-issue C: "Credit Reporting Issues"
      → complaints about incorrect credit report entries, disputes ignored.
    Sub-issue D: "Loan Acquisition & Eligibility"
      → complaints about being denied a loan, eligibility confusion, misleading terms.

{examples_section}
Generate {batch_n} new, fully synthetic complaint narratives about student loans.

Requirements:
- First-person, consumer voice — sometimes frustrated or informal.
- 2–6 sentences each.
- No real names, account numbers, or identifying details.
- Spread the {batch_n} complaints across all four sub-issues as evenly as possible.{topic_hint}

Respond with ONLY a JSON array of {batch_n} objects with exactly these keys:
  "complaint_text" : the narrative text (string)
  "true_issue"     : one of "Loan Servicing & Payments" or "Non-Servicing Issues"
  "true_subissue"  : one of "Loan Information & Servicing", "Payment & Repayment Issues",
                     "Credit Reporting Issues", "Loan Acquisition & Eligibility"

No extra commentary, no markdown, no code fences.
"""

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.choices[0].message.content
        records  = _extract_json_array(raw_text)

        if not records:
            print(f"WARNING: No records parsed for this batch. Raw response:\n{raw_text[:500]}")
            continue

        for rec in records[:batch_n]:
            complaint = str(rec.get("complaint_text", "")).strip()
            if not complaint:
                continue
            all_cleaned_records.append({
                "complaint_text": complaint,
                "true_issue":     str(rec.get("true_issue", "")).strip(),
                "true_subissue":  str(rec.get("true_subissue", "")).strip(),
            })

    return all_cleaned_records[:n]