"""
Advanced ML training: 5-fold stratified CV, hyperparameter tuning,
multiple feature sets, and separate clinical endpoints.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict
)
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings('ignore')

# ─── 1. Load & merge data ─────────────────────────────────────────────────────
ml = pd.read_csv('/Users/acchuaccount/CSS483/data/processed/dataset_ml.csv')

xl = pd.ExcelFile('/Users/acchuaccount/CSS483/data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx')
clin = pd.concat(
    [xl.parse(s) for s in ['HGJ', 'CHUS', 'HMR', 'CHUM']],
    ignore_index=True
)
clin.rename(columns={'Patient #': 'Patient_ID'}, inplace=True)

df = ml.merge(
    clin[['Patient_ID', 'Age', 'Sex', 'Primary Site', 'T-stage', 'N-stage',
          'HPV status', 'Locoregional', 'Distant', 'Death']],
    on='Patient_ID', how='left'
)

# ─── 2. Feature engineering ──────────────────────────────────────────────────
# Sex → binary (handle 'm' mis-entry)
df['Sex'] = df['Sex'].str.strip().str.upper()
df['Sex_M'] = (df['Sex'] == 'M').astype(int)

# Primary Site → is-oropharynx (strongest single-site flag in HN cancer)
df['site_oropharynx'] = (
    df['Primary Site'].str.strip().str.lower() == 'oropharynx'
).astype(int)

# HPV → two-column ternary encoding  (+1/-1 explicit; 0/0 = unknown)
df['hpv_pos'] = (df['HPV status'] == '+').astype(int)
df['hpv_neg'] = (df['HPV status'] == '-').astype(int)

# T-stage → ordinal  (Tx → median imputed)
t_map = {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T4a': 4, 'T4b': 4}
t_vals = df['T-stage'].map(t_map)
df['T_num'] = t_vals.fillna(t_vals.median())

# N-stage → ordinal  (N2a/b/c all → 2)
def _n_ord(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s.startswith('N0'):
        return 0
    if s.startswith('N1'):
        return 1
    if s.startswith('N2'):
        return 2
    if s.startswith('N3'):
        return 3
    return np.nan

n_vals = df['N-stage'].apply(_n_ord)
df['N_num'] = n_vals.fillna(n_vals.median())

# ─── 3. Feature set definitions ──────────────────────────────────────────────
ROI_STATS  = ['roi_mean', 'roi_std', 'roi_p10', 'roi_p90', 'roi_median']
MASK_FEATS = ['mask_pixels', 'mask_roi_overlap']
CLIN_FEATS = ['Age', 'Sex_M', 'site_oropharynx', 'hpv_pos', 'hpv_neg',
              'T_num', 'N_num']

FEAT_SETS = {
    'A: ROI stats':      ROI_STATS,
    'B: ROI + mask':     ROI_STATS + MASK_FEATS,
    'C: ROI + clinical': ROI_STATS + CLIN_FEATS,
    'D: All combined':   ROI_STATS + MASK_FEATS + CLIN_FEATS,
}

# ─── 4. Endpoints ────────────────────────────────────────────────────────────
ENDPOINTS = {
    'LR_or_DM':      'Label',
    'Locoregional':  'Locoregional',
    'Distant':       'Distant',
    'Death':         'Death',
}

# ─── 5. Model factories with GridSearchCV ────────────────────────────────────
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_lr_gs():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=42
        ))
    ])
    return GridSearchCV(
        pipe, {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=CV5, scoring='roc_auc', n_jobs=-1, refit=True
    )

def make_svm_gs():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(class_weight='balanced', probability=True, random_state=42))
    ])
    return GridSearchCV(
        pipe,
        {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']},
        cv=CV5, scoring='roc_auc', n_jobs=-1, refit=True
    )

def make_rf_gs():
    return GridSearchCV(
        RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        {'max_depth': [4, 6, 8],
         'min_samples_leaf': [5, 10],
         'n_estimators': [100]},
        cv=CV5, scoring='roc_auc', n_jobs=-1, refit=True
    )

MODEL_FACTORIES = {
    'LogReg':       make_lr_gs,
    'SVM':          make_svm_gs,
    'RandomForest': make_rf_gs,
}

# ─── 6. Run all combos ───────────────────────────────────────────────────────
results = []
classification_reports = {}

for ep_name, ep_col in ENDPOINTS.items():
    y = df[ep_col].fillna(0).astype(int)
    pos = y.sum()
    print(f"\n{'='*65}")
    print(f"  ENDPOINT: {ep_name:<20}  pos={pos}/{len(y)}  ({pos/len(y)*100:.1f}%)")
    print(f"{'='*65}")

    for fs_name, feats in FEAT_SETS.items():
        X = df[feats].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        for model_name, factory in MODEL_FACTORIES.items():
            gs = factory()
            gs.fit(X, y)
            auc = gs.best_score_
            params = gs.best_params_

            # Cross-val predictions for classification report (outer 5-fold)
            cv_pred = cross_val_predict(
                gs.best_estimator_, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=99),
                method='predict'
            )
            report = classification_report(y, cv_pred, zero_division=0,
                                           target_names=['No event', 'Event'])

            key = (ep_name, fs_name, model_name)
            classification_reports[key] = report

            print(f"  {fs_name:25s} | {model_name:13s} | AUC={auc:.4f} | {params}")
            results.append({
                'Endpoint':   ep_name,
                'FeatureSet': fs_name,
                'Model':      model_name,
                'CV_AUC':     round(auc, 4),
                'BestParams': str(params),
            })

# ─── 7. Save results CSV ─────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv('/tmp/ml_advanced_results.csv', index=False)

# ─── 8. Print pivot tables ───────────────────────────────────────────────────
print("\n\n" + "="*65)
print("  FULL RESULTS TABLE  (5-fold CV ROC-AUC)")
print("="*65)
pivot = res_df.pivot_table(
    index=['Endpoint', 'FeatureSet'],
    columns='Model',
    values='CV_AUC'
)
print(pivot.to_string())

print("\n\nBEST MODEL PER ENDPOINT:")
print("-"*55)
for ep in ENDPOINTS:
    sub = res_df[res_df['Endpoint'] == ep]
    best = sub.loc[sub['CV_AUC'].idxmax()]
    print(f"  {ep:<20} → {best['Model']:13s} | {best['FeatureSet']:25s} | AUC={best['CV_AUC']:.4f}")

# ─── 9. Print classification reports for best config per endpoint ─────────────
print("\n\nCLASSIFICATION REPORTS  (5-fold CV predictions, best feature set per endpoint)")
print("-"*65)
for ep in ENDPOINTS:
    sub = res_df[res_df['Endpoint'] == ep]
    best = sub.loc[sub['CV_AUC'].idxmax()]
    key = (ep, best['FeatureSet'], best['Model'])
    print(f"\n[{ep}] — {best['Model']} / {best['FeatureSet']}")
    print(classification_reports[key])

# ─── 10. Heatmap ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(20, 5))
for ax, (ep_name, _) in zip(axes, ENDPOINTS.items()):
    sub = res_df[res_df['Endpoint'] == ep_name].pivot(
        index='FeatureSet', columns='Model', values='CV_AUC'
    )
    im = ax.imshow(sub.values, vmin=0.45, vmax=0.85, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=25, ha='right', fontsize=8)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels([s.split(':')[0] for s in sub.index], fontsize=9)
    for r in range(len(sub.index)):
        for c in range(len(sub.columns)):
            val = sub.values[r, c]
            ax.text(c, r, f"{val:.3f}", ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='black' if 0.55 < val < 0.80 else 'white')
    ax.set_title(ep_name, fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('5-Fold CV ROC-AUC  |  Feature Set × Model × Endpoint', fontsize=12)
plt.tight_layout()
plt.savefig('/tmp/ml_advanced_heatmap.png', dpi=150, bbox_inches='tight')
print("\nHeatmap saved: /tmp/ml_advanced_heatmap.png")

# ─── 11. Feature importance bar chart (best RF model, all endpoints) ──────────
fig2, axes2 = plt.subplots(1, len(ENDPOINTS), figsize=(20, 5))
for ax, (ep_name, ep_col) in zip(axes2, ENDPOINTS.items()):
    y = df[ep_col].fillna(0).astype(int)
    # find best feature set for RF on this endpoint
    sub = res_df[(res_df['Endpoint'] == ep_name) & (res_df['Model'] == 'RandomForest')]
    if sub.empty:
        continue
    best_fs = sub.loc[sub['CV_AUC'].idxmax(), 'FeatureSet']
    feats = FEAT_SETS[best_fs]
    X = df[feats].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    rf_gs = make_rf_gs()
    rf_gs.fit(X, y)
    importances = rf_gs.best_estimator_.feature_importances_
    order = np.argsort(importances)[::-1]
    ax.barh(range(len(feats)), importances[order][::-1], color='steelblue')
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([feats[i] for i in order[::-1]], fontsize=7)
    ax.set_title(f"{ep_name}\n({best_fs})", fontsize=8)
    ax.set_xlabel('Importance', fontsize=7)

plt.suptitle('RF Feature Importances by Endpoint (best feature set)', fontsize=11)
plt.tight_layout()
plt.savefig('/tmp/ml_rf_importances_by_endpoint.png', dpi=150, bbox_inches='tight')
print("RF importances chart saved: /tmp/ml_rf_importances_by_endpoint.png")
print("\nResults CSV saved: /tmp/ml_advanced_results.csv")
