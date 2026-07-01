"""
Trains the hierarchical (2-level) student loan complaint classifier, the
same architecture used for evaluation in model_evaluation.py and saves all
fitted artifacts to a single joblib bundle, so the Streamlit app can load a
ready-to-use pipeline without retraining on every run.

Expected project layout (run this script once from anywhere, paths are
resolved relative to the repo root):

    <repo_root>/
        data/
            X_train_tfidf.npz
            tfidf_vectorizer.pkl
            student_loan_augmented.csv
        app/
            train_and_save_models.py   <- this file

Usage:
    python app/train_and_save_models.py

Output:
    data/hierarchical_model_bundle.pkl
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Level 1 (broad) groups -> same mapping used in model_evaluation.py
GROUPING = {
    "Credit Reporting Issues": "Non-Servicing Issues",
    "Loan Acquisition & Eligibility": "Non-Servicing Issues",
    "Loan Information & Servicing": "Loan Servicing & Payments",
    "Payment & Repayment Issues": "Loan Servicing & Payments",
}

# Defined once, candidate grid reused by Level 1 LR and SVC GridSearchCV calls.
LEVEL1_C_GRID = [0.1, 1.0, 5.0, 10.0, 20.0]

# Defined once, candidate grid reused by both LR and SVC GridSearchCV calls per group below.
LEVEL2_C_GRID = [0.1, 1.0, 5.0, 10.0, 20.0]


def main():
    print("Loading TF-IDF features and training data...")
    X_train = load_npz(os.path.join(DATA_DIR, "X_train_tfidf.npz"))
    train_df = pd.read_csv(os.path.join(DATA_DIR, "student_loan_augmented.csv"))

    y_train_subissue = train_df["Subissue_grouped"]
    y_train_broad = y_train_subissue.map(GROUPING)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # Level 1: broad issue (LogisticRegression + calibrated LinearSVC)
    # Grid-searched, same as Level 2, instead of trusting a single fixed C.
    # ------------------------------------------------------------------
    print("\n[Level 1] Grid-searching C for LogisticRegression and calibrated LinearSVC...")

    lr_broad_gs = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        param_grid={"C": LEVEL1_C_GRID}, cv=cv, scoring="f1_macro", n_jobs=-1,
    )
    lr_broad_gs.fit(X_train, y_train_broad)
    print(f"   [LR]  Best C: {lr_broad_gs.best_params_['C']}  |  CV Macro F1: {lr_broad_gs.best_score_:.4f}")
    lr_broad = lr_broad_gs.best_estimator_
    lr_broad_c = lr_broad_gs.best_params_["C"]

    svc_broad_gs = GridSearchCV(
        CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=1000, random_state=42),
            cv=3, method="isotonic",
        ),
        param_grid={"estimator__C": LEVEL1_C_GRID}, cv=cv, scoring="f1_macro", n_jobs=-1,
    )
    svc_broad_gs.fit(X_train, y_train_broad)
    print(
        f"   [SVC] Best C: {svc_broad_gs.best_params_['estimator__C']}  "
        f"|  CV Macro F1: {svc_broad_gs.best_score_:.4f}"
    )
    svc_broad = svc_broad_gs.best_estimator_
    svc_broad_c = svc_broad_gs.best_params_["estimator__C"]

    broad_classes = lr_broad.classes_
    print(f"   Broad classes: {list(broad_classes)}")

    # Out-of-fold broad-issue probabilities, used as cascade features for Level 2.
    # Reuses the C values just found above, instead of refitting another grid search.
    print("\n[Level 1] Computing out-of-fold probabilities for feature cascading...")
    oof_lr = cross_val_predict(
        LogisticRegression(C=lr_broad_c, class_weight="balanced", max_iter=1000, random_state=42),
        X_train, y_train_broad, cv=cv, method="predict_proba", n_jobs=-1,
    )
    oof_svc = cross_val_predict(
        CalibratedClassifierCV(
            LinearSVC(C=svc_broad_c, class_weight="balanced", max_iter=1000, random_state=42),
            cv=3, method="isotonic",
        ),
        X_train, y_train_broad, cv=cv, method="predict_proba", n_jobs=-1,
    )
    train_broad_proba = (oof_lr + oof_svc) / 2.0

    train_df = train_df.copy()
    train_df["broad_group"] = y_train_broad.values

    # ------------------------------------------------------------------
    # Level 2: sub-issue classifiers per broad group, with cascaded
    # broad-issue probabilities appended to the TF-IDF features
    # ------------------------------------------------------------------
    print("\n[Level 2] Grid-searching C and fitting sub-issue classifiers per broad group...")
    level2_models = {}
    for broad_group in broad_classes:
        train_mask = train_df["broad_group"] == broad_group
        grp_train_df = train_df[train_mask].copy()
        grp_train_broad_proba = train_broad_proba[train_mask.values]

        X_train_tfidf = X_train[train_mask.values]
        broad_proba_sp = csr_matrix(grp_train_broad_proba.astype(np.float32))
        X_train_group = hstack([X_train_tfidf, broad_proba_sp]).tocsr()
        y_train_subgroup = grp_train_df["Subissue_grouped"]

        print(f"\n---> Group: '{broad_group}' ({X_train_group.shape[0]} samples)")

        lr_sub = GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            param_grid={"C": LEVEL2_C_GRID}, cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        lr_sub.fit(X_train_group, y_train_subgroup)
        print(f"     [LR]  Best C: {lr_sub.best_params_['C']}  |  CV Macro F1: {lr_sub.best_score_:.4f}")
        lr_sub = lr_sub.best_estimator_

        min_class_count = y_train_subgroup.value_counts().min()
        safe_cv = min(3, min_class_count)
        svc_sub = GridSearchCV(
            CalibratedClassifierCV(
                LinearSVC(max_iter=1000, class_weight="balanced", random_state=42),
                cv=safe_cv, method="isotonic",
            ),
            param_grid={"estimator__C": LEVEL2_C_GRID}, cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        svc_sub.fit(X_train_group, y_train_subgroup)
        print(
            f"     [SVC] Best C: {svc_sub.best_params_['estimator__C']}  "
            f"|  CV Macro F1: {svc_sub.best_score_:.4f}"
        )
        svc_sub = svc_sub.best_estimator_

        # 'Non-Servicing Issues' is a small group (1879 samples). Its grid search
        # repeatedly lands on the grid's upper edge (5.0 -> 20.0 -> 75.0 across
        # widened grids) for marginal CV F1 gains (~0.002-0.003), which is plateau
        # noise, not real signal. We deliberately override to a less-aggressive
        # C=10.0 for both models here to reduce overfitting risk on this small
        # group, rather than chase the grid's edge indefinitely.
        if broad_group == "Non-Servicing Issues":
            override_c = 10.0
            print(f"     [Override] Small group plateau: forcing C={override_c} for LR and SVC.")
            lr_sub = LogisticRegression(
                C=override_c, max_iter=1000, class_weight="balanced", random_state=42
            )
            lr_sub.fit(X_train_group, y_train_subgroup)
            svc_sub = CalibratedClassifierCV(
                LinearSVC(C=override_c, max_iter=1000, class_weight="balanced", random_state=42),
                cv=safe_cv, method="isotonic",
            )
            svc_sub.fit(X_train_group, y_train_subgroup)

        level2_models[broad_group] = {
            "lr": lr_sub,
            "svc": svc_sub,
            "sub_classes": lr_sub.classes_,
        }
        print(
            f"     -> '{broad_group}': classes: {list(lr_sub.classes_)}"
        )

    # ------------------------------------------------------------------
    # Save everything needed for inference
    # ------------------------------------------------------------------
    bundle = {
        "lr_broad": lr_broad,
        "svc_broad": svc_broad,
        "broad_classes": broad_classes,
        "level2_models": level2_models,
        "grouping": GROUPING,
    }

    out_path = os.path.join(DATA_DIR, "hierarchical_model_bundle.pkl")
    joblib.dump(bundle, out_path)
    print(f"\nSaved trained pipeline bundle to: {out_path}")


if __name__ == "__main__":
    main()