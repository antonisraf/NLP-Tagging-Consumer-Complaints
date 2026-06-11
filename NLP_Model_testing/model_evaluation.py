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

# Δημιουργία φακέλου για τα plots αν δεν υπάρχει
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

lr_pipe = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
                       param_grid={'C': [1.0]}, cv=cv, scoring='f1_macro', n_jobs=-1)
lr_pipe.fit(X_train, y_train_issue_broad)
# We use CalibratedClassifierCV with LinearSVC to get probability estimates for the SVC model
svc_pipe = GridSearchCV(CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=1000, random_state=42), cv=3, method='isotonic'),
                        param_grid={'estimator__C': [1.0]}, cv=cv, scoring='f1_macro', n_jobs=-1)
svc_pipe.fit(X_train, y_train_issue_broad)
# Level 1 predictions and probabilities for the test set
proba_lr_test  = lr_pipe.best_estimator_.predict_proba(X_test)
proba_svc_test = svc_pipe.best_estimator_.predict_proba(X_test)
broad_classes  = lr_pipe.best_estimator_.classes_
# We average the probabilities from both models to get a more robust confidence score for the broad issue classification
avg_proba_broad  = (proba_lr_test + proba_svc_test) / 2.0
y_pred_broad     = broad_classes[np.argmax(avg_proba_broad, axis=1)]
level1_confidence = np.max(avg_proba_broad, axis=1)
# For the training set, we also need the out-of-fold probabilities to use as features for the level 2 sub-issue classifiers
oof_lr = cross_val_predict(LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42), X_train, y_train_issue_broad, cv=cv, method='predict_proba', n_jobs=-1)
oof_svc = cross_val_predict(CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=1000, random_state=42), cv=3, method='isotonic'), X_train, y_train_issue_broad, cv=cv, method='predict_proba', n_jobs=-1)
train_broad_proba = (oof_lr + oof_svc) / 2.0

train_df['broad_group'] = y_train_issue_broad.values
final_subissue_preds      = np.empty(len(test_df), dtype=object)
final_subissue_confidence = np.zeros(len(test_df), dtype=float)

# We will train separate sub-issue classifiers for each broad issue category and apply them to the corresponding subsets of the test data
for broad_group in broad_classes:
    train_mask = train_df['broad_group'] == broad_group
    grp_train_df = train_df[train_mask].copy()
    grp_train_broad_proba = train_broad_proba[train_mask.values]

    X_train_tfidf   = tfidf.transform(grp_train_df['cleaned_text'])
    broad_proba_sp  = csr_matrix(grp_train_broad_proba.astype(np.float32))
    X_train_group   = hstack([X_train_tfidf, broad_proba_sp])
    y_train_subgroup  = grp_train_df['Subissue_grouped']

    test_mask = y_pred_broad == broad_group

    lr_sub = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
    lr_sub.fit(X_train_group, y_train_subgroup)
    
    svc_sub = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=1000, class_weight='balanced', random_state=42), cv=3, method='isotonic')
    svc_sub.fit(X_train_group, y_train_subgroup)

    if test_mask.sum() > 0:
        X_test_tfidf  = X_test[test_mask]
        test_bp_sp    = csr_matrix(avg_proba_broad[test_mask].astype(np.float32))
        X_test_group  = hstack([X_test_tfidf, test_bp_sp])

        avg_sub_proba = (lr_sub.predict_proba(X_test_group) + svc_sub.predict_proba(X_test_group)) / 2.0
        sub_classes = lr_sub.classes_
        
        final_subissue_preds[test_mask] = sub_classes[np.argmax(avg_sub_proba, axis=1)]
        final_subissue_confidence[test_mask] = level1_confidence[test_mask] * np.max(avg_sub_proba, axis=1)

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
