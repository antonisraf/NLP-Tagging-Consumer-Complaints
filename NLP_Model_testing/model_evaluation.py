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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Create plot folder
os.makedirs('plots', exist_ok=True)

# Set matplotlib style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

# Loading data and preparing pipelines

X_train = load_npz('data/X_train_tfidf.npz')
X_test  = load_npz('data/X_test_tfidf.npz')
train_df = pd.read_csv('data/student_loan_augmented.csv')
test_df  = pd.read_csv('data/student_loan_test.csv')
tfidf    = joblib.load('data/tfidf_vectorizer.pkl')

# After analysis we decidedto group the issues into 2 broad categories
GROUPING = {
    'Credit Reporting Issues':        'Non-Servicing Issues',
    'Loan Acquisition & Eligibility': 'Non-Servicing Issues',
    'Loan Information & Servicing':   'Loan Servicing & Payments',
    'Payment & Repayment Issues':     'Loan Servicing & Payments',
}

y_train_subissue = train_df['Subissue_grouped']
y_test_subissue  = test_df['Subissue_grouped']
y_train_issue_broad = y_train_subissue.map(GROUPING)
y_test_issue_broad  = y_test_subissue.map(GROUPING)

# Configuring StratifiedKFold for consistent cross-validation splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# C values confirmed via live GridSearchCV in train_and_save_models.py
# (reproduced identically across 3 separate runs on the same X_train_tfidf.npz).
# Kept in sync here so evaluation measures the same model that's actually deployed.
LEVEL1_LR_C = 5.0
LEVEL1_SVC_C = 1.0

# Loan Servicing & Payments: grid search result, used as-is.
# Non-Servicing Issues: grid search landed on 20.0 (grid edge, plateau ~0.002-0.003
# F1 gain from 5.0 onward), overridden to 10.0 for lower overfitting risk on this
# small group (1879 samples). See train_and_save_models.py for the full reasoning.
LEVEL2_C = {
    'Loan Servicing & Payments': {'lr': 1.0, 'svc': 0.1},
    'Non-Servicing Issues':      {'lr': 10.0, 'svc': 10.0},
}


lr_model = LogisticRegression(C=LEVEL1_LR_C, class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train_issue_broad)

# 2. Calibrated LinearSVC 
svc_model = CalibratedClassifierCV(
    LinearSVC(C=LEVEL1_SVC_C, class_weight='balanced', max_iter=1000, random_state=42), 
    cv=3, 
    method='isotonic'
)
svc_model.fit(X_train, y_train_issue_broad)

# Level 1 predictions and probabilities for the test set
proba_lr_test  = lr_model.predict_proba(X_test)
proba_svc_test = svc_model.predict_proba(X_test)
broad_classes  = lr_model.classes_
# We average the probabilities from both models to get a more robust confidence score for the broad issue classification
avg_proba_broad  = (proba_lr_test + proba_svc_test) / 2.0
y_pred_broad     = broad_classes[np.argmax(avg_proba_broad, axis=1)]
level1_confidence = np.max(avg_proba_broad, axis=1)
# For the training set, we also need the out-of-fold probabilities to use as features for the level 2 sub-issue classifiers
oof_lr = cross_val_predict(LogisticRegression(C=LEVEL1_LR_C, class_weight='balanced', max_iter=1000, random_state=42), X_train, y_train_issue_broad, cv=cv, method='predict_proba', n_jobs=-1)
oof_svc = cross_val_predict(CalibratedClassifierCV(LinearSVC(C=LEVEL1_SVC_C, class_weight='balanced', max_iter=1000, random_state=42), cv=3, method='isotonic'), X_train, y_train_issue_broad, cv=cv, method='predict_proba', n_jobs=-1)
train_broad_proba = (oof_lr + oof_svc) / 2.0

train_df['broad_group'] = y_train_issue_broad.values
final_subissue_preds      = np.empty(len(test_df), dtype=object)
final_subissue_confidence = np.zeros(len(test_df), dtype=float)
l2_entropy                = np.zeros(len(test_df), dtype=float)

# We will train separate sub-issue classifiers for each broad issue category and apply them to the corresponding subsets of the test data
for broad_group in broad_classes:
    train_mask = train_df['broad_group'] == broad_group
    grp_train_df = train_df[train_mask].copy()
    grp_train_broad_proba = train_broad_proba[train_mask.values]

    X_train_tfidf   = X_train[train_mask.values]
    broad_proba_sp = csr_matrix(grp_train_broad_proba.astype(np.float32))
    X_train_group   = hstack([X_train_tfidf, broad_proba_sp])
    y_train_subgroup  = grp_train_df['Subissue_grouped']

    test_mask = y_pred_broad == broad_group

    lr_sub = LogisticRegression(C=LEVEL2_C[broad_group]['lr'], max_iter=1000, class_weight='balanced', random_state=42)
    lr_sub.fit(X_train_group, y_train_subgroup)
    
    svc_sub = CalibratedClassifierCV(LinearSVC(C=LEVEL2_C[broad_group]['svc'], max_iter=1000, class_weight='balanced', random_state=42), cv=3, method='isotonic')
    svc_sub.fit(X_train_group, y_train_subgroup)

    if test_mask.sum() > 0:
        X_test_tfidf  = X_test[test_mask]
        test_bp_sp = csr_matrix(avg_proba_broad[test_mask].astype(np.float32))
        X_test_group  = hstack([X_test_tfidf, test_bp_sp])

        avg_sub_proba = (lr_sub.predict_proba(X_test_group) + svc_sub.predict_proba(X_test_group)) / 2.0
        sub_classes = lr_sub.classes_
        
        final_subissue_preds[test_mask] = sub_classes[np.argmax(avg_sub_proba, axis=1)]
        final_subissue_confidence[test_mask] = level1_confidence[test_mask] * np.max(avg_sub_proba, axis=1)
        l2_entropy[test_mask] = -np.sum(avg_sub_proba * np.log(avg_sub_proba + 1e-10), axis=1)

# Automated evaluation across a range of thresholds to analyze the trade-off between automation and human review
thresholds = np.arange(0.30, 0.86, 0.05)
plot_thresholds, plot_review_pct, plot_auto_f1 = [], [], []

for t in thresholds:
    auto_mask = final_subissue_confidence >= t
    n_auto = auto_mask.sum()
    pct_review = ((len(test_df) - n_auto) / len(test_df)) * 100
    
    if n_auto > 0:
        auto_f1 = f1_score(y_test_subissue[auto_mask], final_subissue_preds[auto_mask], average='macro')
    else:
        auto_f1 = 0.0
    
    plot_thresholds.append(t)
    plot_review_pct.append(pct_review)
    plot_auto_f1.append(auto_f1)

# Dashboard generation

PRIMARY_COLOR = '#1e3d59'    # Deep Navy
SECONDARY_COLOR = '#ff6e40'  # Soft Coral
THRESHOLD_COLOR = '#222222'  # Black Dashed Line

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

fig.suptitle('NLP Hierarchical Model: Operational & Threshold Analysis', 
             fontsize=16, weight='bold', color='#111111', y=0.98)

# Subplot 1: The Automation Trade-off
ax1 = axes[0]
ax1.set_title('Automation Optimization Trade-Off', fontsize=13, weight='bold', pad=15, color='#333333')

ax1.set_xlabel('Rejection Threshold', labelpad=10)
ax1.set_ylabel('Auto Subset Macro F1', color=PRIMARY_COLOR, labelpad=10, weight='bold')
line1 = ax1.plot(plot_thresholds, plot_auto_f1, color=PRIMARY_COLOR, marker='o', 
                 linewidth=2.5, markersize=6, label='Auto Subset Macro F1')
ax1.tick_params(axis='y', labelcolor=PRIMARY_COLOR)
ax1.grid(True, linestyle='--', alpha=0.5)

ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('Human Review Rate (%)', color=SECONDARY_COLOR, labelpad=15, weight='bold', rotation=270)
line2 = ax1_twin.plot(plot_thresholds, plot_review_pct, color=SECONDARY_COLOR, marker='s', 
                     linewidth=2.5, markersize=6, linestyle=':', label='Human Review %')
ax1_twin.tick_params(axis='y', labelcolor=SECONDARY_COLOR)
ax1_twin.grid(False)

v_line = ax1.axvline(x=0.45, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.5, alpha=0.9, label='Selected Threshold (0.45)')

lines = line1 + line2 + [v_line]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), 
           ncol=1, fontsize=9.5, frameon=True, facecolor='white', edgecolor='none')

# Subplot 2: Confidence Distribution Histogram
ax2 = axes[1]
ax2.set_title('Joint Prediction Confidence Distribution', fontsize=13, weight='bold', pad=15, color='#333333')

sns.histplot(final_subissue_confidence, bins=25, kde=True, color=PRIMARY_COLOR, alpha=0.4, ax=ax2, edgecolor='white')
ax2.axvline(x=0.45, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.5, alpha=0.9)

ax2.text(0.45, ax2.get_ylim()[1] * 0.90, 'Threshold: 0.45', color=THRESHOLD_COLOR, 
         weight='bold', fontsize=10, ha='center',
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))

ax2.set_xlabel('Joint Probability Score $P(L1) \\times P(L2)$', labelpad=10)
ax2.set_ylabel('Count of Complaints', labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.5)

# Subplot 3: Operational Donut Chart (Breakdown at 0.45)
ax3 = axes[2]
chosen_threshold = 0.45
auto_mask_chosen = final_subissue_confidence >= chosen_threshold
n_auto_total = auto_mask_chosen.sum()
n_review_total = len(test_df) - n_auto_total
sizes = [n_auto_total, n_review_total]

ax3.set_title(f'Workflow Allocation (at Threshold = {chosen_threshold})', fontsize=13, weight='bold', pad=15, color='#333333')

wedges, texts, autotexts = ax3.pie(
    sizes, 
    labels=['Auto-labelled', 'Human Review'], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=[PRIMARY_COLOR, SECONDARY_COLOR], 
    pctdistance=0.75,
    textprops=dict(color='#111111', fontsize=10)
)

plt.setp(autotexts, size=11, weight="bold", color="white")

centre_circle = plt.Circle((0,0), 0.55, fc='white', linewidth=0)
fig.gca().add_artist(centre_circle)

ax3.text(0, -0.05, f"Total\n{len(test_df)}", ha='center', va='center', fontsize=11, weight='bold', color='#444444')

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.35, top=0.82, bottom=0.22)
plt.savefig('plots/nlp_performance_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# Final report and confusion matrices at the chosen threshold for the auto-labelled subset
print("\n" + "=" * 60)
print("Final evaluation at chosen threshold:")
print("=" * 60)

print("\n--- Level 1 (Broad Issue, 2-class) ---")
print(classification_report(y_test_issue_broad, y_pred_broad))

print("-> Confusion Matrix level 1 (Rows: True, Cols: Predicted):")
cm_l1 = confusion_matrix(y_test_issue_broad, y_pred_broad, labels=broad_classes)
df_cm_l1 = pd.DataFrame(cm_l1, index=broad_classes, columns=broad_classes)
print(df_cm_l1.to_string())

print(f"\n--- Sub-issue (Level 2) — Auto-labelled only (Confidence >= {chosen_threshold}) ---")
print(classification_report(y_test_subissue[auto_mask_chosen], final_subissue_preds[auto_mask_chosen]))

print("-> Confusion Matrix level 2 (Auto-labelled Only) (Rows: True, Cols: Predicted):")
sub_classes_unique = sorted(y_test_subissue.unique())
cm_l2 = confusion_matrix(y_test_subissue[auto_mask_chosen], final_subissue_preds[auto_mask_chosen], labels=sub_classes_unique)
df_cm_l2 = pd.DataFrame(cm_l2, index=sub_classes_unique, columns=sub_classes_unique)
print(df_cm_l2.to_string())

# ======================================================================
# Joint Perplexity on real test data, and its correlation with error
# ======================================================================
# Mirrors the metric computed in app/model_pipeline.py's predict(), but here
# on the real labelled test set instead of synthetic Streamlit complaints.
# This is NOT the same as final_subissue_confidence: confidence is the max
# probability, perplexity is the entropy-derived spread of the whole
# distribution. Two predictions can have identical top-1 confidence but very
# different perplexity if the remaining mass is concentrated in one runner-up
# vs. spread across many classes.
from scipy import stats
from sklearn.metrics import roc_auc_score

l1_entropy = -np.sum(avg_proba_broad * np.log(avg_proba_broad + 1e-10), axis=1)
joint_perplexity = np.exp(l1_entropy + l2_entropy)

# Error at the sub-issue level (final prediction, before any threshold routing)
is_error = (final_subissue_preds != y_test_subissue.values).astype(int)

print("\n" + "=" * 60)
print("Joint Perplexity vs. Prediction Error (real test data)")
print("=" * 60)

print(f"\nJoint Perplexity — mean: {joint_perplexity.mean():.3f}, "
      f"median: {np.median(joint_perplexity):.3f}, "
      f"std: {joint_perplexity.std():.3f}")
print(f"Overall sub-issue error rate: {is_error.mean():.3%}")

mean_pp_correct = joint_perplexity[is_error == 0].mean()
mean_pp_wrong   = joint_perplexity[is_error == 1].mean()
print(f"\nMean perplexity | correct predictions: {mean_pp_correct:.3f}")
print(f"Mean perplexity | wrong predictions:    {mean_pp_wrong:.3f}")

# Point-biserial correlation: continuous perplexity vs. binary error.
# Captures linear association and its direction.
pb_corr, pb_pvalue = stats.pointbiserialr(is_error, joint_perplexity)
print(f"\nPoint-biserial correlation (perplexity, error): r = {pb_corr:.3f}  (p = {pb_pvalue:.4g})")

# AUC-ROC: how well perplexity alone ranks/separates correct vs. wrong
# predictions. More appropriate than r here since we only care about
# "higher perplexity -> more likely wrong", not a linear relationship.
# 0.5 = no better than random, 1.0 = perfect separation.
auc = roc_auc_score(is_error, joint_perplexity)
print(f"AUC-ROC (perplexity predicting error):        {auc:.3f}")

if pb_corr > 0 and auc >= 0.70:
    strength = "strong and in the expected direction"
elif pb_corr > 0 and auc >= 0.60:
    strength = "moderate and in the expected direction"
elif pb_corr > 0:
    strength = "weak but in the expected direction"
else:
    strength = "absent or inverted — investigate before trusting this metric for routing"
print(f"\nInterpretation: relationship is {strength}.")

# Perplexity distribution plot: correct vs. wrong, saved alongside the main dashboard
fig2, ax = plt.subplots(figsize=(7, 5))
sns.histplot(joint_perplexity[is_error == 0], bins=25, color=PRIMARY_COLOR, alpha=0.5,
             label='Correct', kde=True, ax=ax, stat='density')
sns.histplot(joint_perplexity[is_error == 1], bins=25, color=SECONDARY_COLOR, alpha=0.5,
             label='Wrong', kde=True, ax=ax, stat='density')
ax.set_title(f'Joint Perplexity by Outcome (r={pb_corr:.2f}, AUC={auc:.2f})',
             fontsize=13, weight='bold', color='#333333')
ax.set_xlabel('Joint Perplexity  =  exp(H(L1) + H(L2))', labelpad=10)
ax.set_ylabel('Density', labelpad=10)
ax.legend(frameon=True, facecolor='white', edgecolor='none')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/joint_perplexity_vs_error.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: plots/joint_perplexity_vs_error.png")

# ----------------------------------------------------------------------
# Does perplexity actually catch the DOMINANT confusion pair, or just
# catch easier/rarer errors while missing the main one?
# ----------------------------------------------------------------------
# The two sub-issues below share vocabulary and get confused with each
# other far more than with anything else (see confusion matrix above).
# Aggregate AUC/r can look "fine" while being blind to exactly this pair,
# if the model is confidently wrong there (sharp distribution, wrong peak)
# rather than genuinely uncertain (flat distribution).
CONFUSABLE_PAIR = {'Loan Information & Servicing', 'Payment & Repayment Issues'}

is_confusable_pair_error = (
    (is_error == 1)
    & y_test_subissue.isin(CONFUSABLE_PAIR).values
    & pd.Series(final_subissue_preds).isin(CONFUSABLE_PAIR).values
)
is_other_error = (is_error == 1) & (~is_confusable_pair_error)
is_correct = is_error == 0

print("\n" + "=" * 60)
print("Perplexity broken down by error type")
print("=" * 60)
print(f"\nCorrect predictions:              n={is_correct.sum():5d}  mean perplexity={joint_perplexity[is_correct].mean():.3f}")
print(f"Confusable-pair errors (Info<->Payment): n={is_confusable_pair_error.sum():5d}  mean perplexity={joint_perplexity[is_confusable_pair_error].mean():.3f}")
print(f"Other errors:                      n={is_other_error.sum():5d}  mean perplexity={joint_perplexity[is_other_error].mean():.3f}")

# AUC of perplexity separating correct vs. confusable-pair errors specifically
mask_pair_vs_correct = is_correct | is_confusable_pair_error
if is_confusable_pair_error.sum() > 0:
    auc_pair = roc_auc_score(
        is_confusable_pair_error[mask_pair_vs_correct].astype(int),
        joint_perplexity[mask_pair_vs_correct]
    )
    print(f"\nAUC (perplexity separating confusable-pair errors from correct): {auc_pair:.3f}")

# AUC of perplexity separating correct vs. all other (non-pair) errors
mask_other_vs_correct = is_correct | is_other_error
if is_other_error.sum() > 0:
    auc_other = roc_auc_score(
        is_other_error[mask_other_vs_correct].astype(int),
        joint_perplexity[mask_other_vs_correct]
    )
    print(f"AUC (perplexity separating other errors from correct):           {auc_other:.3f}")

print(
    "\nIf AUC(confusable-pair) << AUC(other), the metric is mostly catching\n"
    "the easier/rarer errors and is blind to the main failure mode driving\n"
    "review volume, which means it should NOT be trusted alone to route\n"
    "these specific complaints away from human review."
)