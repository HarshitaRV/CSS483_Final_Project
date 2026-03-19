import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/acchuaccount/CSS483'

# -------------------------
# Table I: fold composition
# -------------------------
rows = []
current = None
pat = re.compile(r'^(\w+)_?\w* positives=(\d+) negatives=(\d+)$')
fold_pat = re.compile(r'^\s*fold(\d+): n=(\d+) pos=(\d+) neg=(\d+)$')

with open('/tmp/fold_counts.log', 'r') as f:
    for line in f:
        line = line.strip()
        m = pat.match(line)
        if m:
            current = m.group(1)
            continue
        fm = fold_pat.match(line)
        if fm and current is not None:
            rows.append({
                'Endpoint': current,
                'Fold': int(fm.group(1)),
                'Fold_N': int(fm.group(2)),
                'Fold_Pos': int(fm.group(3)),
                'Fold_Neg': int(fm.group(4)),
            })

table1 = pd.DataFrame(rows)
table1.to_csv('/tmp/Table_I_fold_composition.csv', index=False)

# ---------------------------------------
# Table II: baseline holdout metrics parse
# ---------------------------------------
baseline_rows = []
model = None
auc = None
acc = None
f1_event = None

with open('/tmp/train_ml_baseline.log', 'r') as f:
    for raw in f:
        line = raw.rstrip('\n')
        if 'Logistic Regression' in line:
            model = 'Logistic Regression'
            acc = None
            f1_event = None
        elif line.strip() == 'SVM':
            model = 'SVM'
            acc = None
            f1_event = None
        elif 'Random Forest' in line and line.strip() == 'Random Forest':
            model = 'Random Forest'
            acc = None
            f1_event = None
        elif line.strip().startswith('accuracy'):
            parts = line.split()
            # expected ... accuracy ... value support
            try:
                acc = float(parts[-2])
            except Exception:
                pass
        elif 'Tumor failure (1)' in line:
            parts = line.split()
            try:
                # precision recall f1 support in columns
                f1_event = float(parts[-2])
            except Exception:
                pass
        elif line.startswith('ROC-AUC:') and model is not None:
            auc = float(line.split(':')[1].strip())
            baseline_rows.append({
                'Model': model,
                'Accuracy': acc,
                'F1_Event': f1_event,
                'ROC_AUC': auc,
            })
            model = None

table2 = pd.DataFrame(baseline_rows)
table2.to_csv('/tmp/Table_II_baseline_holdout_metrics.csv', index=False)

# ------------------------------------------------------
# Table III: cross-validated tuned metrics endpoint/set
# ------------------------------------------------------
metrics = pd.read_csv('/tmp/model_feature_endpoint_cv_metrics.csv')
metrics = metrics.sort_values(['Endpoint', 'FeatureSet', 'Model'])
metrics.to_csv('/tmp/Table_III_cv_tuned_metrics.csv', index=False)

# -----------------------------------
# Table IV: subgroup AUC/F1 summary
# -----------------------------------
sub = pd.read_csv('/tmp/ml_subgroup_results.csv')
sub = sub.sort_values(['Endpoint', 'SubgroupVar', 'Level'])
sub.to_csv('/tmp/Table_IV_subgroup_auc_f1.csv', index=False)

# -----------------------------------
# Fig 4: feature distributions
# -----------------------------------
df = pd.read_csv(f'{BASE}/dataset_ml.csv')
features = ['mask_pixels', 'mask_roi_overlap', 'roi_mean', 'roi_std', 'roi_p10', 'roi_p90', 'roi_median']

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.ravel()

for i, feat in enumerate(features):
    ax = axes[i]
    vals = df[feat].dropna().values
    ax.hist(vals, bins=30, color='#4C78A8', alpha=0.85, edgecolor='white')
    ax.set_title(feat)
    ax.set_xlabel('')
    ax.set_ylabel('Count')

# hide unused axes
for j in range(len(features), len(axes)):
    axes[j].axis('off')

plt.suptitle('Feature Distributions: Sarcopenia-related and ROI Statistics', fontsize=14)
plt.tight_layout()
plt.savefig('/tmp/Fig_4_feature_distributions.png', dpi=170, bbox_inches='tight')

print('Saved /tmp/Table_I_fold_composition.csv')
print('Saved /tmp/Table_II_baseline_holdout_metrics.csv')
print('Saved /tmp/Table_III_cv_tuned_metrics.csv')
print('Saved /tmp/Table_IV_subgroup_auc_f1.csv')
print('Saved /tmp/Fig_4_feature_distributions.png')
