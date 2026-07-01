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
from sklearn.preprocessing import normalize

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

        # Complexity scoring artifacts, fit once on the training set in
        # train_and_save_models.py. Independent of lr_broad/svc_broad/level2
        # confidence on purpose, see note in that script.
        self.complexity_centroids = bundle.get("complexity_centroids")
        self.complexity_count_vectorizer = bundle.get("complexity_count_vectorizer")
        self.complexity_lda_model = bundle.get("complexity_lda_model")

    def _compute_complexity_scores(self, X_tfidf, cleaned_texts):
        """
        Per-complaint complexity, computed independently of the classifier's
        own predict_proba output:

        - centroid_margin_ambiguity: 1 - (similarity to closest class
          centroid - similarity to 2nd closest), in TF-IDF cosine space.
          High = text sits semantically between two sub-issue classes.
        - topic_entropy: normalized entropy of the LDA topic distribution
          (fit on raw counts, no labels involved at all). High = text is
          topically scattered rather than concentrated in one theme.

        Returns two 1-D numpy arrays aligned with the input rows.
        """
        n = X_tfidf.shape[0]

        # --- centroid margin ambiguity ---
        if self.complexity_centroids:
            classes = list(self.complexity_centroids.keys())
            centroid_matrix = np.vstack([self.complexity_centroids[c] for c in classes])
            X_norm = normalize(X_tfidf, norm="l2", axis=1)
            sims = X_norm.dot(centroid_matrix.T)  # (n_docs, n_classes)
            sims = np.asarray(sims.todense()) if hasattr(sims, "todense") else np.asarray(sims)
            sims_sorted = np.sort(sims, axis=1)[:, ::-1]
            top1 = sims_sorted[:, 0]
            top2 = sims_sorted[:, 1] if sims_sorted.shape[1] > 1 else np.zeros(n)
            margin = top1 - top2
            centroid_margin_ambiguity = 1.0 - margin
        else:
            centroid_margin_ambiguity = np.full(n, np.nan)

        # --- LDA topic entropy ---
        if self.complexity_count_vectorizer is not None and self.complexity_lda_model is not None:
            X_counts = self.complexity_count_vectorizer.transform(cleaned_texts)
            topic_dist = self.complexity_lda_model.transform(X_counts)
            eps = 1e-12
            raw_entropy = -np.sum(topic_dist * np.log(topic_dist + eps), axis=1)
            max_entropy = np.log(topic_dist.shape[1])
            topic_entropy = raw_entropy / max_entropy if max_entropy > 0 else raw_entropy
        else:
            topic_entropy = np.full(n, np.nan)

        return centroid_margin_ambiguity, topic_entropy

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

        centroid_margin_ambiguity, topic_entropy = self._compute_complexity_scores(X, cleaned)

        return pd.DataFrame({
            "complaint_text": texts,
            "cleaned_text": cleaned,
            "predicted_issue_broad": y_pred_broad,
            "issue_confidence": level1_confidence,
            "predicted_subissue": sub_preds,
            "subissue_confidence": sub_confidence,
            "joint_confidence": joint_confidence,
            "needs_review": joint_confidence < threshold,
            "complexity_centroid_margin": centroid_margin_ambiguity,
            "complexity_topic_entropy": topic_entropy,
        })

def apply_eval_review_override(results: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluation-only routing override.

    In real production use there is no ground-truth label, so `needs_review`
    is decided purely by `joint_confidence < threshold` inside `predict()`.

    In evaluation/demo contexts (e.g. the Streamlit app, which generates
    synthetic complaints with a known `true_issue`), we additionally route a
    complaint to human review whenever the predicted broad issue (L1) is
    wrong, even if the model was confident. This catches high-confidence L1
    mistakes that the joint-confidence threshold alone would miss.

    Must be called AFTER `issue_correct` has been computed (i.e. after
    comparing `predicted_issue_broad` to `true_issue`). Kept separate from
    `predict()` because `predict()` has no access to ground truth.
    """
    if "issue_correct" not in results.columns:
        raise ValueError(
            "apply_eval_review_override requires an 'issue_correct' column. "
            "Compute it first (predicted_issue_broad == true_issue)."
        )

    out = results.copy()
    out["needs_review"] = out["needs_review"] | (~out["issue_correct"])
    return out
