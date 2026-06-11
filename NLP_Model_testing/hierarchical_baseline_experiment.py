import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

from NLP_Model_training.utils_NLP import back_translate_dataframe, clean_tfidf_text

# Rejection threshold for auto-labeling
REJECTION_THRESHOLD = 0.45

# I regroup the original 4 Issue classes into 2 broad groups based on confusion patterns:
GROUPING_ORIGINAL = {
    'Credit Reporting Issues':         'Credit Reporting Issues',
    'Loan Acquisition & Eligibility':  'Loan Acquisition & Eligibility',
    'Loan Information & Servicing':    'Loan Information & Servicing',
    'Payment & Repayment Issues':      'Payment & Repayment Issues',
}

GROUPING_ALTERNATIVE = {
    'Credit Reporting Issues':         'Non-Servicing Issues',
    'Loan Acquisition & Eligibility':  'Non-Servicing Issues',
    'Loan Information & Servicing':    'Loan Servicing & Payments',
    'Payment & Repayment Issues':      'Loan Servicing & Payments',
}

# Load preprocessed data and vectorizer
print("1. Loading saved data (50k Sparse TF-IDF)...")

X_train = load_npz('data/X_train_tfidf.npz')
X_test  = load_npz('data/X_test_tfidf.npz')

train_df = pd.read_csv('data/student_loan_augmented.csv')
test_df  = pd.read_csv('data/student_loan_test.csv')
tfidf    = joblib.load('data/tfidf_vectorizer.pkl')

y_train_issue   = train_df['Issue_grouped']
y_test_issue    = test_df['Issue_grouped']
y_test_subissue = test_df['Subissue_grouped']

print(f"Train X: {X_train.shape} | Test X: {X_test.shape}")


# Run the full pipeline (Level 1 + Level 2) and return all relevant outputs for analysis
def run_pipeline(X_train, X_test, train_df, test_df, tfidf,
                 y_train_issue, y_test_issue, y_test_subissue,
                 label="original"):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Level 1 ---
    lr_pipe = GridSearchCV(
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        param_grid={'C': [0.01, 0.1, 0.5, 1.0]},
        cv=cv, scoring='f1_macro', n_jobs=-1, refit=True
    )
    lr_pipe.fit(X_train, y_train_issue)
    best_lr = lr_pipe.best_estimator_

    svc_pipe = GridSearchCV(
        CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=1000, random_state=42),
            cv=3, method='isotonic'
        ),
        param_grid={'estimator__C': [0.01, 0.1, 0.5, 1.0]},
        cv=cv, scoring='f1_macro', n_jobs=-1, refit=True
    )
    svc_pipe.fit(X_train, y_train_issue)
    best_svc = svc_pipe.best_estimator_

    proba_lr_test  = best_lr.predict_proba(X_test)
    proba_svc_test = best_svc.predict_proba(X_test)
    issue_classes  = best_lr.classes_

    avg_proba_issue = (proba_lr_test + proba_svc_test) / 2.0
    y_pred_issue    = issue_classes[np.argmax(avg_proba_issue, axis=1)]
    test_issue_proba = avg_proba_issue

    # Out-of-fold probabilities for cascade
    oof_lr  = cross_val_predict(
        LogisticRegression(C=best_lr.C, class_weight='balanced',
                           max_iter=1000, random_state=42),
        X_train, y_train_issue, cv=cv, method='predict_proba', n_jobs=-1
    )
    oof_svc = cross_val_predict(
        CalibratedClassifierCV(
            LinearSVC(C=best_svc.estimator.C, class_weight='balanced',
                      max_iter=1000, random_state=42),
            cv=3, method='isotonic'
        ),
        X_train, y_train_issue, cv=cv, method='predict_proba', n_jobs=-1
    )
    train_issue_proba = (oof_lr + oof_svc) / 2.0

    # --- Level 2 ---
    final_subissue_preds      = np.empty(len(test_df), dtype=object)
    final_subissue_confidence = np.zeros(len(test_df), dtype=float)
    unique_issues             = train_df['Issue_grouped'].unique()

    for issue_group in unique_issues:
        train_mask            = train_df['Issue_grouped'] == issue_group
        grp_train_df          = train_df[train_mask].copy()
        grp_train_issue_proba = train_issue_proba[train_mask.values]

        X_train_tfidf  = tfidf.transform(grp_train_df['cleaned_text'])
        issue_proba_sp = csr_matrix(grp_train_issue_proba.astype(np.float32))
        X_train_group  = hstack([X_train_tfidf, issue_proba_sp])

        y_train_subgroup  = grp_train_df['Subissue_grouped']
        unique_subclasses = y_train_subgroup.unique()
        test_mask         = y_pred_issue == issue_group

        if len(unique_subclasses) == 1:
            if test_mask.sum() > 0:
                final_subissue_preds[test_mask]      = unique_subclasses[0]
                final_subissue_confidence[test_mask] = 1.0
            continue

        
        lr_sub = GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            param_grid={'C': [0.1, 1.0, 5.0]}, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        lr_sub.fit(X_train_group, y_train_subgroup)

        min_class_count = y_train_subgroup.value_counts().min()
        use_svc = min_class_count >= 2
        if use_svc:
            safe_cv = min(3, min_class_count)
            
            svc_sub = GridSearchCV(
                CalibratedClassifierCV(
                    LinearSVC(max_iter=1000, class_weight='balanced', random_state=42),
                    cv=safe_cv, method='isotonic'
                ),
                param_grid={'estimator__C': [0.1, 1.0, 5.0]}, cv=cv, scoring='f1_macro', n_jobs=-1
            )
            svc_sub.fit(X_train_group, y_train_subgroup)

        if test_mask.sum() > 0:
            X_test_tfidf = X_test[test_mask]
            test_ip_sp   = csr_matrix(test_issue_proba[test_mask].astype(np.float32))
            X_test_group = hstack([X_test_tfidf, test_ip_sp])

            if use_svc:
                avg_sub_proba = (
                    lr_sub.best_estimator_.predict_proba(X_test_group)
                    + svc_sub.best_estimator_.predict_proba(X_test_group)
                ) / 2.0
            else:
                avg_sub_proba = lr_sub.best_estimator_.predict_proba(X_test_group)

            sub_classes = lr_sub.best_estimator_.classes_
            final_subissue_preds[test_mask]      = sub_classes[np.argmax(avg_sub_proba, axis=1)]
            final_subissue_confidence[test_mask] = avg_sub_proba.max(axis=1)

    issue_macro_f1 = f1_score(y_test_issue, y_pred_issue, average='macro')
    sub_macro_f1   = f1_score(y_test_subissue, final_subissue_preds, average='macro')

    return (issue_macro_f1, sub_macro_f1,
            y_pred_issue, avg_proba_issue,
            final_subissue_preds, final_subissue_confidence,
            issue_classes, train_issue_proba)


# Original grouping pipeline run
print("\n" + "=" * 60)
print("2. Training Level 1 — Issue Classifier (original 4-class grouping)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("-> Grid-searching C for LogisticRegression (Level 1)...")
lr_pipe = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    param_grid={'C': [0.01, 0.1, 0.5, 1.0]},
    cv=cv, scoring='f1_macro', n_jobs=-1, refit=True
)
lr_pipe.fit(X_train, y_train_issue)
best_lr_issue = lr_pipe.best_estimator_
print(f"   Best C (LR): {lr_pipe.best_params_['C']}  |  CV Macro F1: {lr_pipe.best_score_:.4f}")

print("-> Grid-searching C for CalibratedLinearSVC (Level 1)...")
svc_pipe = GridSearchCV(
    CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=1000, random_state=42),
        cv=3, method='isotonic'
    ),
    param_grid={'estimator__C': [0.01, 0.1, 0.5, 1.0]},
    cv=cv, scoring='f1_macro', n_jobs=-1, refit=True
)
svc_pipe.fit(X_train, y_train_issue)
best_svc_issue = svc_pipe.best_estimator_
print(f"   Best C (SVC): {svc_pipe.best_params_['estimator__C']}  |  CV Macro F1: {svc_pipe.best_score_:.4f}")

print("-> Computing soft-vote predictions for Level 1...")
proba_lr_test  = best_lr_issue.predict_proba(X_test)
proba_svc_test = best_svc_issue.predict_proba(X_test)
issue_classes  = best_lr_issue.classes_

avg_proba_issue  = (proba_lr_test + proba_svc_test) / 2.0
y_pred_issue     = issue_classes[np.argmax(avg_proba_issue, axis=1)]
test_issue_proba = avg_proba_issue

print("-> Computing out-of-fold Issue probabilities for cascade training...")
oof_lr  = cross_val_predict(
    LogisticRegression(C=best_lr_issue.C, class_weight='balanced',
                       max_iter=1000, random_state=42),
    X_train, y_train_issue, cv=cv, method='predict_proba', n_jobs=-1
)
oof_svc = cross_val_predict(
    CalibratedClassifierCV(
        LinearSVC(C=best_svc_issue.estimator.C, class_weight='balanced',
                  max_iter=1000, random_state=42),
        cv=3, method='isotonic'
    ),
    X_train, y_train_issue, cv=cv, method='predict_proba', n_jobs=-1
)
train_issue_proba = (oof_lr + oof_svc) / 2.0
print("   Out-of-fold probabilities ready.")

# Sub-issue classifiers with feature cascading
print("\n" + "=" * 60)
print("3. Training Level 2 — Sub-issue Classifiers (with feature cascading)")
print("=" * 60)

final_subissue_preds      = np.empty(len(test_df), dtype=object)
final_subissue_confidence = np.zeros(len(test_df), dtype=float)
unique_issues             = train_df['Issue_grouped'].unique()

for issue_group in unique_issues:
    print(f"\n---> Group: '{issue_group}'")

    train_mask            = train_df['Issue_grouped'] == issue_group
    grp_train_df          = train_df[train_mask].copy()
    grp_train_issue_proba = train_issue_proba[train_mask.values]

    X_train_tfidf  = tfidf.transform(grp_train_df['cleaned_text'])
    issue_proba_sp = csr_matrix(grp_train_issue_proba.astype(np.float32))
    X_train_group  = hstack([X_train_tfidf, issue_proba_sp])

    y_train_subgroup  = grp_train_df['Subissue_grouped']
    unique_subclasses = y_train_subgroup.unique()
    test_mask         = y_pred_issue == issue_group

    if len(unique_subclasses) == 1:
        print(f"     [Info] Single class ('{unique_subclasses[0]}'). Assigning directly.")
        if test_mask.sum() > 0:
            final_subissue_preds[test_mask]      = unique_subclasses[0]
            final_subissue_confidence[test_mask] = 1.0
        continue

    print(f"     [Train] Soft-vote Ensemble (LR + calibrated SVC) | "
          f"{len(unique_subclasses)} classes | {X_train_group.shape[0]} samples")
    print(f"     [Distribution]\n{y_train_subgroup.value_counts().to_string()}")

    
    lr_sub = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        param_grid={'C': [0.1, 1.0, 5.0]}, cv=cv, scoring='f1_macro', n_jobs=-1
    )
    lr_sub.fit(X_train_group, y_train_subgroup)
    print(f"     [LR]  Best C: {lr_sub.best_params_['C']}  |  CV Macro F1: {lr_sub.best_score_:.4f}")

    min_class_count = y_train_subgroup.value_counts().min()
    use_svc = min_class_count >= 2
    if use_svc:
        safe_cv = min(3, min_class_count)
        if safe_cv < 3:
            print(f"     [Calib] min_class_count={min_class_count} — reducing CV to {safe_cv} folds.")
      
        svc_sub = GridSearchCV(
            CalibratedClassifierCV(
                LinearSVC(max_iter=1000, class_weight='balanced', random_state=42),
                cv=safe_cv, method='isotonic'
            ),
            param_grid={'estimator__C': [0.1, 1.0, 5.0]}, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        svc_sub.fit(X_train_group, y_train_subgroup)
        print(f"     [SVC] Best C: {svc_sub.best_params_['estimator__C']}  |  CV Macro F1: {svc_sub.best_score_:.4f}")
    else:
        print(f"     [Calib] min_class_count={min_class_count} — SVC skipped, using LR only.")

    if test_mask.sum() > 0:
        X_test_tfidf = X_test[test_mask]
        test_ip_sp   = csr_matrix(test_issue_proba[test_mask].astype(np.float32))
        X_test_group = hstack([X_test_tfidf, test_ip_sp])

        if use_svc:
            avg_sub_proba = (
                lr_sub.best_estimator_.predict_proba(X_test_group)
                + svc_sub.best_estimator_.predict_proba(X_test_group)
            ) / 2.0
        else:
            avg_sub_proba = lr_sub.best_estimator_.predict_proba(X_test_group)

        sub_classes = lr_sub.best_estimator_.classes_
        final_subissue_preds[test_mask]      = sub_classes[np.argmax(avg_sub_proba, axis=1)]
        final_subissue_confidence[test_mask] = avg_sub_proba.max(axis=1)

# Confusion matrix analysis for Level 1 predictions
print("\n" + "=" * 60)
print("4. Confusion Matrix Analysis (Level 1 — Issue)")
print("=" * 60)

cm     = confusion_matrix(y_test_issue, y_pred_issue, labels=issue_classes)
cm_df  = pd.DataFrame(cm, index=issue_classes, columns=issue_classes)
print("\nConfusion matrix (rows=true, cols=predicted):")
print(cm_df.to_string())

# Off-diagonal confusion rates between each pair
print("\nTop confusion pairs (% of true class misclassified as other):")
confusion_pairs = []
for i, true_cls in enumerate(issue_classes):
    row_total = cm[i].sum()
    for j, pred_cls in enumerate(issue_classes):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'True':      true_cls,
                'Predicted': pred_cls,
                'Count':     cm[i, j],
                'Rate':      cm[i, j] / row_total
            })
cp_df = (pd.DataFrame(confusion_pairs)
           .sort_values('Rate', ascending=False)
           .head(10)
           .reset_index(drop=True))
cp_df['Rate'] = cp_df['Rate'].map('{:.1%}'.format)
print(cp_df.to_string(index=False))

# Alternative grouping pipeline run
print("\n" + "=" * 60)
print("5. Testing Alternative Grouping (2 broad Issue classes)")
print("=" * 60)

# Apply alternative grouping to labels
y_train_issue_alt = y_train_issue.map(GROUPING_ALTERNATIVE)
y_test_issue_alt  = y_test_issue.map(GROUPING_ALTERNATIVE)

cv_alt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("-> Grid-searching C for LogisticRegression (alt grouping)...")
lr_alt = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    param_grid={'C': [0.01, 0.1, 0.5, 1.0]},
    cv=cv_alt, scoring='f1_macro', n_jobs=-1, refit=True
)
lr_alt.fit(X_train, y_train_issue_alt)
best_lr_alt = lr_alt.best_estimator_
print(f"   Best C (LR): {lr_alt.best_params_['C']}  |  CV Macro F1: {lr_alt.best_score_:.4f}")

print("-> Grid-searching C for CalibratedLinearSVC (alt grouping)...")
svc_alt = GridSearchCV(
    CalibratedClassifierCV(
        LinearSVC(class_weight='balanced', max_iter=1000, random_state=42),
        cv=3, method='isotonic'
    ),
    param_grid={'estimator__C': [0.01, 0.1, 0.5, 1.0]},
    cv=cv_alt, scoring='f1_macro', n_jobs=-1, refit=True
)
svc_alt.fit(X_train, y_train_issue_alt)
best_svc_alt = svc_alt.best_estimator_
print(f"   Best C (SVC): {svc_alt.best_params_['estimator__C']}  |  CV Macro F1: {svc_alt.best_score_:.4f}")

proba_lr_alt  = best_lr_alt.predict_proba(X_test)
proba_svc_alt = best_svc_alt.predict_proba(X_test)
avg_proba_alt = (proba_lr_alt + proba_svc_alt) / 2.0
issue_classes_alt = best_lr_alt.classes_
y_pred_issue_alt  = issue_classes_alt[np.argmax(avg_proba_alt, axis=1)]

issue_macro_f1_alt = f1_score(y_test_issue_alt, y_pred_issue_alt, average='macro')
print(f"\n--- Alt Grouping Level 1 ---")
print(classification_report(y_test_issue_alt, y_pred_issue_alt))

# Threshold-based rejection analysis on sub-issue predictions
print("\n" + "=" * 60)
print(f"6. Threshold-Based Rejection  (threshold = {REJECTION_THRESHOLD})")
print("=" * 60)

# Split into auto-labelled vs human-review
auto_mask   = final_subissue_confidence >= REJECTION_THRESHOLD
review_mask = ~auto_mask

n_total  = len(test_df)
n_auto   = auto_mask.sum()
n_review = review_mask.sum()

print(f"\n   Total predictions : {n_total}")
print(f"   Auto-labelled     : {n_auto}  ({n_auto/n_total:.1%})")
print(f"   Sent for review   : {n_review}  ({n_review/n_total:.1%})")

# Evaluate only on auto-labelled subset
auto_sub_f1 = f1_score(
    y_test_subissue[auto_mask],
    final_subissue_preds[auto_mask],
    average='macro'
)
print(f"\n   Sub-issue Macro F1 on AUTO subset : {auto_sub_f1:.4f}")
print(f"   Sub-issue Macro F1 on ALL         : "
      f"{f1_score(y_test_subissue, final_subissue_preds, average='macro'):.4f}")

# Distribution of rejected samples by true label
review_true_labels = y_test_subissue[review_mask]
print(f"\n   Review pile — true label distribution:")
print(review_true_labels.value_counts().to_string())

# Confidence histogram buckets
bins   = [0.0, 0.40, 0.55, 0.70, 0.85, 1.01]
labels = ['<0.40', '0.40-0.55', '0.55-0.70', '0.70-0.85', '>0.85']
conf_series = pd.Series(final_subissue_confidence)
bucketed    = pd.cut(conf_series, bins=bins, labels=labels, right=False)
print(f"\n   Confidence distribution across ALL predictions:")
print(bucketed.value_counts().sort_index().to_string())

# Final evaluation summary
print("\n" + "=" * 60)
print("7. Final Evaluation Summary")
print("=" * 60)

print("\n--- Issue (Level 1) — Original 4-class grouping ---")
print(classification_report(y_test_issue, y_pred_issue))
issue_macro_f1 = f1_score(y_test_issue, y_pred_issue, average='macro')

print("\n--- Sub-issue (Level 2) — All predictions ---")
print(classification_report(y_test_subissue, final_subissue_preds))
sub_macro_f1 = f1_score(y_test_subissue, final_subissue_preds, average='macro')

print("\n--- Sub-issue (Level 2) — Auto-labelled only (above threshold) ---")
print(classification_report(
    y_test_subissue[auto_mask],
    final_subissue_preds[auto_mask]
))

print("\n" + "=" * 60)
print(f"  Issue     Macro F1  (original grouping)    : {issue_macro_f1:.4f}")
print(f"  Issue     Macro F1  (alt 2-class grouping) : {issue_macro_f1_alt:.4f}")
print(f"  Sub-issue Macro F1  (all predictions)      : {sub_macro_f1:.4f}")
print(f"  Sub-issue Macro F1  (auto-labelled only)   : {auto_sub_f1:.4f}")
print(f"  Sent for human review                      : {n_review}/{n_total} ({n_review/n_total:.1%})")
print("=" * 60)