"""
Inference-only wrapper around the hierarchical complaint classifier.

Loads:
  - data/tfidf_vectorizer.pkl   (fitted in vectorizer.py)
  - data/hierarchical_model_bundle.pkl  (fitted in train_and_save_models.py)

and exposes `HierarchicalComplaintClassifier.predict(texts)`, which mirrors
the Level1 -> Level2 cascade logic used in model_evaluation.py.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Make the existing NLP_Model_training package importable so we reuse the
# exact same text-cleaning function used at training time.
sys.path.append(BASE_DIR)
from NLP_Model_training.utils_NLP import clean_tfidf_text  # noqa: E402

# Threshold below which a prediction is routed to "human review" instead of
# being auto-labelled (same default used in model_evaluation.py / the
# hierarchical_baseline_experiment dashboards).
DEFAULT_REJECTION_THRESHOLD = 0.45


class HierarchicalComplaintClassifier:
    def __init__(self, data_dir=DATA_DIR):
        tfidf_path = os.path.join(data_dir, "tfidf_vectorizer.pkl")
        bundle_path = os.path.join(data_dir, "hierarchical_model_bundle.pkl")

        if not os.path.exists(tfidf_path):
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {tfidf_path}. "
                "Run vectorizer.py first."
            )
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. "
                "Run app/train_and_save_models.py first."
            )

        self.tfidf = joblib.load(tfidf_path)
        bundle = joblib.load(bundle_path)
        self.lr_broad = bundle["lr_broad"]
        self.svc_broad = bundle["svc_broad"]
        self.broad_classes = bundle["broad_classes"]
        self.level2_models = bundle["level2_models"]

    def predict(self, texts, threshold=DEFAULT_REJECTION_THRESHOLD):
        """
        Parameters
        ----------
        texts : list[str]
            Raw complaint narratives (uncleaned).
        threshold : float
            Joint-confidence cutoff used to flag rows for human review.

        Returns
        -------
        pandas.DataFrame with one row per input text and columns:
            complaint_text, cleaned_text,
            predicted_issue_broad, issue_confidence,
            predicted_subissue, subissue_confidence,
            joint_confidence, needs_review
        """
        cleaned = [clean_tfidf_text(t) for t in texts]
        X = self.tfidf.transform(cleaned)

        # --- Level 1: broad issue ---
        proba_lr = self.lr_broad.predict_proba(X)
        proba_svc = self.svc_broad.predict_proba(X)
        avg_proba_broad = (proba_lr + proba_svc) / 2.0
        y_pred_broad = self.broad_classes[np.argmax(avg_proba_broad, axis=1)]
        level1_confidence = np.max(avg_proba_broad, axis=1)

        # --- Level 2: sub-issue, per predicted broad group ---
        n = X.shape[0]
        sub_preds = np.empty(n, dtype=object)
        sub_confidence = np.zeros(n, dtype=float)

        for broad_group in self.broad_classes:
            mask = y_pred_broad == broad_group
            if not mask.any():
                continue

            models = self.level2_models[broad_group]
            X_sub = X[mask]
            proba_sp = csr_matrix(avg_proba_broad[mask].astype(np.float32))
            X_group = hstack([X_sub, proba_sp]).tocsr()

            avg_sub_proba = (
                models["lr"].predict_proba(X_group)
                + models["svc"].predict_proba(X_group)
            ) / 2.0
            sub_classes = models["sub_classes"]

            sub_preds[mask] = sub_classes[np.argmax(avg_sub_proba, axis=1)]
            sub_confidence[mask] = level1_confidence[mask] * np.max(avg_sub_proba, axis=1)

        joint_confidence = sub_confidence

        return pd.DataFrame({
            "complaint_text": texts,
            "cleaned_text": cleaned,
            "predicted_issue_broad": y_pred_broad,
            "issue_confidence": level1_confidence,
            "predicted_subissue": sub_preds,
            "subissue_confidence": sub_confidence,
            "joint_confidence": joint_confidence,
            "needs_review": joint_confidence < threshold,
        })
