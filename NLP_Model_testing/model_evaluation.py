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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from scipy.stats import mannwhitneyu, pointbiserialr
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

# ------------------------------------------------------------------
# Complexity scoring (independent of the classifiers' own confidence)
#
# Fit only on train_df, applied to test_df. Neither score touches
# predict_proba from lr_model/svc_model/lr_sub/svc_sub, the goal is to
# measure how inherently ambiguous/mixed the TEXT is, not how unsure the
# classifier happens to be. Using classifier confidence here would just be
# circular (low confidence "explaining" low confidence).
# ------------------------------------------------------------------
print("\n=== Fitting complexity scorers (centroid margin + LDA topic entropy) ===")

# --- 1. Centroid margin ambiguity (TF-IDF cosine space) ---
centroid_classes = sorted(train_df['Subissue_grouped'].unique())
X_train_norm = normalize(X_train, norm='l2', axis=1)

# Global centroid = mean over ALL docs, the vocabulary shared across every
# class (loan, payment, account, call...). Raw class centroids are dominated
# by this shared signal (cosine sim between classes was ~0.95, nearly
# constant score, no discriminative power). Subtracting it isolates what's
# actually distinctive per class before computing the margin.
global_centroid = np.asarray(X_train_norm.mean(axis=0)).ravel()

centroids = {}
for cls in centroid_classes:
    mask = (train_df['Subissue_grouped'] == cls).values
    raw_centroid = np.asarray(X_train_norm[mask].mean(axis=0)).ravel()
    centered = raw_centroid - global_centroid
    norm_val = np.linalg.norm(centered)
    centroids[cls] = centered / norm_val if norm_val > 0 else centered

centroid_matrix = np.vstack([centroids[c] for c in centroid_classes])
X_test_norm = normalize(X_test, norm='l2', axis=1)
test_sims = np.asarray(X_test_norm.dot(centroid_matrix.T))
test_sims_sorted = np.sort(test_sims, axis=1)[:, ::-1]
test_centroid_margin_ambiguity = 1.0 - (test_sims_sorted[:, 0] - test_sims_sorted[:, 1])

# --- 2. LDA topic entropy (raw counts, no labels involved) ---
count_vectorizer = CountVectorizer(max_features=5000, min_df=3, max_df=0.95)
X_train_counts = count_vectorizer.fit_transform(train_df['cleaned_text'])
X_test_counts  = count_vectorizer.transform(test_df['cleaned_text'])

lda_n_topics = 8
lda_model = LatentDirichletAllocation(n_components=lda_n_topics, random_state=42, learning_method='online', n_jobs=-1)
lda_model.fit(X_train_counts)

test_topic_dist = lda_model.transform(X_test_counts)
_eps = 1e-12
test_raw_entropy = -np.sum(test_topic_dist * np.log(test_topic_dist + _eps), axis=1)
test_topic_entropy = test_raw_entropy / np.log(lda_n_topics)

test_df = test_df.copy()
test_df['complexity_centroid_margin'] = test_centroid_margin_ambiguity
test_df['complexity_topic_entropy']   = test_topic_entropy

print(f"  Centroid margin ambiguity — mean: {test_centroid_margin_ambiguity.mean():.3f}, std: {test_centroid_margin_ambiguity.std():.3f}")
print(f"  Topic entropy             — mean: {test_topic_entropy.mean():.3f}, std: {test_topic_entropy.std():.3f}")

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

# ------------------------------------------------------------------
# Complexity score analysis
#
# Two separate questions, deliberately kept separate:
#
# 1. Does the review queue actually contain harder text? (group comparison
#    of complexity scores: needs_review vs auto-labelled, at chosen_threshold)
# 2. Does text complexity predict wrong sub-issue predictions, independent
#    of whether the classifier flagged it for review? (correlation with
#    subissue_correct, across the full test set)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("Complexity score analysis (independent of classifier confidence)")
print("=" * 60)

subissue_correct = (final_subissue_preds == y_test_subissue.values)
needs_review_chosen = ~auto_mask_chosen

for score_name in ['complexity_centroid_margin', 'complexity_topic_entropy']:
    scores = test_df[score_name].values

    review_scores = scores[needs_review_chosen]
    auto_scores   = scores[auto_mask_chosen]
    u_stat, p_value = mannwhitneyu(review_scores, auto_scores, alternative='greater')

    corr, corr_p = pointbiserialr(subissue_correct.astype(int), scores)

    print(f"\n--- {score_name} ---")
    print(f"  Mean (human review group): {review_scores.mean():.3f}")
    print(f"  Mean (auto-labelled group): {auto_scores.mean():.3f}")
    print(f"  Mann-Whitney U (review > auto): p = {p_value:.4g}")
    print(f"  Point-biserial correlation with subissue_correct: r = {corr:.3f} (p = {corr_p:.4g})")

# Plot: complexity score distributions by review status, and error rate by decile
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.suptitle('Complexity Score Diagnostics (Independent of Classifier Confidence)',
              fontsize=14, weight='bold', color='#111111', y=0.98)

score_to_plot = 'complexity_centroid_margin'

ax_box = axes2[0]
box_data = pd.DataFrame({
    score_to_plot: test_df[score_to_plot].values,
    'Routed to': np.where(needs_review_chosen, 'Human Review', 'Auto-labelled'),
})
sns.boxplot(data=box_data, x='Routed to', y=score_to_plot, ax=ax_box,
            palette=[PRIMARY_COLOR, SECONDARY_COLOR])
ax_box.set_title('Centroid Ambiguity by Routing Outcome', fontsize=12, weight='bold', pad=12)
ax_box.set_ylabel('Centroid Margin Ambiguity (higher = more ambiguous)', labelpad=10)
ax_box.set_xlabel('')
ax_box.grid(True, linestyle='--', alpha=0.5)

ax_err = axes2[1]
decile_labels = pd.qcut(test_df[score_to_plot], q=5, duplicates='drop')
err_by_decile = pd.DataFrame({
    'decile': decile_labels,
    'correct': subissue_correct,
}).groupby('decile', observed=True)['correct'].agg(['mean', 'count'])
err_rate = 1 - err_by_decile['mean']
ax_err.bar(range(len(err_rate)), err_rate.values, color=PRIMARY_COLOR, alpha=0.8)
ax_err.set_xticks(range(len(err_rate)))
ax_err.set_xticklabels([f"Q{i+1}" for i in range(len(err_rate))])
ax_err.set_title('Sub-issue Error Rate by Ambiguity Quintile', fontsize=12, weight='bold', pad=12)
ax_err.set_xlabel('Centroid Ambiguity Quintile (Q1=lowest, Q5=highest)', labelpad=10)
ax_err.set_ylabel('Error Rate', labelpad=10)
ax_err.grid(True, linestyle='--', alpha=0.5)

plt.subplots_adjust(left=0.07, right=0.96, wspace=0.3, top=0.85, bottom=0.12)
plt.savefig('plots/complexity_score_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved complexity diagnostics plot to: plots/complexity_score_diagnostics.png")