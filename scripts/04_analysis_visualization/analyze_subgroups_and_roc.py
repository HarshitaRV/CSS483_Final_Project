"""Exploratory subgroup analysis + ROC visualization using tuned model configs."""

import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

RESULTS_CSV = '/tmp/ml_advanced_results.csv'
OUT_SUMMARY = '/tmp/ml_auc_f1_by_endpoint_featureset.csv'
OUT_SUBGROUP = '/tmp/ml_subgroup_results.csv'
OUT_DRIVERS = '/tmp/ml_feature_drivers.csv'
OUT_IMAGING = '/tmp/ml_imaging_value_summary.csv'
OUT_ROC = '/tmp/ml_roc_by_endpoint.png'
OUT_HEAT_AUC = '/tmp/ml_auc_heatmap_best_model_per_featureset.png'
OUT_HEAT_F1 = '/tmp/ml_f1_heatmap_best_model_per_featureset.png'
OUT_DRIVERS_PLOT = '/tmp/ml_feature_drivers_top10.png'


def load_data():
    ml = pd.read_csv('/Users/acchuaccount/CSS483/data/processed/dataset_ml.csv')
    xl = pd.ExcelFile('/Users/acchuaccount/CSS483/data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx')
    clin = pd.concat([xl.parse(s) for s in ['HGJ', 'CHUS', 'HMR', 'CHUM']], ignore_index=True)
    clin.rename(columns={'Patient #': 'Patient_ID'}, inplace=True)

    df = ml.merge(
        clin[['Patient_ID', 'Age', 'Sex', 'Primary Site', 'T-stage', 'N-stage',
              'HPV status', 'Locoregional', 'Distant', 'Death']],
        on='Patient_ID', how='left'
    )

    df['Sex'] = df['Sex'].astype(str).str.strip().str.upper()
    df['Sex_M'] = (df['Sex'] == 'M').astype(int)
    df['site_oropharynx'] = (df['Primary Site'].astype(str).str.strip().str.lower() == 'oropharynx').astype(int)
    df['hpv_pos'] = (df['HPV status'] == '+').astype(int)
    df['hpv_neg'] = (df['HPV status'] == '-').astype(int)

    t_map = {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T4A': 4, 'T4B': 4}
    t_vals = df['T-stage'].astype(str).str.upper().map(t_map)
    df['T_num'] = t_vals.fillna(t_vals.median())

    def n_ord(v):
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

    n_vals = df['N-stage'].apply(n_ord)
    df['N_num'] = n_vals.fillna(n_vals.median())

    # Exploratory proxy: lower mask area can indicate low muscle burden.
    q1 = df['mask_pixels'].quantile(0.25)
    df['sarcopenia_proxy'] = np.where(df['mask_pixels'] <= q1, 'Low area (Q1)', 'Higher area (Q2-4)')

    # Subgroup buckets for exploratory performance stratification.
    df['Age_group'] = np.where(df['Age'] >= 65, '>=65', '<65')
    df['Sex_group'] = np.where(df['Sex'] == 'F', 'F', 'M')
    df['HPV_group'] = np.where(df['HPV status'] == '+', 'HPV+', np.where(df['HPV status'] == '-', 'HPV-', 'Unknown'))
    df['T_group'] = np.where(df['T_num'] >= 3, 'T3-4', 'T1-2')
    df['N_group'] = np.where(df['N_num'] >= 2, 'N2-3', 'N0-1')

    return df


ROI_STATS = ['roi_mean', 'roi_std', 'roi_p10', 'roi_p90', 'roi_median']
MASK_FEATS = ['mask_pixels', 'mask_roi_overlap']
CLIN_FEATS = ['Age', 'Sex_M', 'site_oropharynx', 'hpv_pos', 'hpv_neg', 'T_num', 'N_num']

FEAT_SETS = {
    'A: ROI stats': ROI_STATS,
    'B: ROI + mask': ROI_STATS + MASK_FEATS,
    'C: ROI + clinical': ROI_STATS + CLIN_FEATS,
    'D: All combined': ROI_STATS + MASK_FEATS + CLIN_FEATS,
}

ENDPOINTS = {
    'LR_or_DM': 'Label',
    'Locoregional': 'Locoregional',
    'Distant': 'Distant',
    'Death': 'Death',
}

SUBGROUP_VARS = ['Age_group', 'Sex_group', 'HPV_group', 'T_group', 'N_group', 'sarcopenia_proxy']



def build_estimator(model_name, params):
    if model_name == 'LogReg':
        c = params.get('clf__C', 1.0)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42))
        ])

    if model_name == 'SVM':
        c = params.get('clf__C', 1.0)
        kernel = params.get('clf__kernel', 'rbf')
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=c, kernel=kernel, class_weight='balanced', probability=True, random_state=42))
        ])

    return RandomForestClassifier(
        max_depth=params.get('max_depth', 6),
        min_samples_leaf=params.get('min_samples_leaf', 5),
        n_estimators=params.get('n_estimators', 100),
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )



def get_importances(fitted_estimator, model_name, X, y, feature_names):
    if model_name == 'RandomForest' and hasattr(fitted_estimator, 'feature_importances_'):
        vals = fitted_estimator.feature_importances_
    elif model_name in ('LogReg', 'SVM') and isinstance(fitted_estimator, Pipeline):
        clf = fitted_estimator.named_steps['clf']
        if hasattr(clf, 'coef_'):
            vals = np.abs(clf.coef_).ravel()
        else:
            perm = permutation_importance(
                fitted_estimator, X, y,
                n_repeats=20, random_state=42,
                scoring='roc_auc'
            )
            vals = perm.importances_mean
    else:
        perm = permutation_importance(
            fitted_estimator, X, y,
            n_repeats=20, random_state=42,
            scoring='roc_auc'
        )
        vals = perm.importances_mean

    imp = pd.DataFrame({'feature': feature_names, 'importance': vals})
    return imp.sort_values('importance', ascending=False).reset_index(drop=True)



def safe_auc(y_true, y_score):
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))



def main():
    df = load_data()
    tuned = pd.read_csv(RESULTS_CSV)

    # Pick best model for each endpoint+feature set from prior tuning.
    best_per_pair = (
        tuned.sort_values('CV_AUC', ascending=False)
        .groupby(['Endpoint', 'FeatureSet'], as_index=False)
        .first()
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

    summary_rows = []
    pred_store = {}

    print('Running CV predictions for best model of each endpoint + feature set...')
    for _, row in best_per_pair.iterrows():
        endpoint = row['Endpoint']
        fs = row['FeatureSet']
        model_name = row['Model']
        params = ast.literal_eval(row['BestParams'])

        y = df[ENDPOINTS[endpoint]].fillna(0).astype(int)
        X = df[FEAT_SETS[fs]].copy()
        for c in X.columns:
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median())

        est = build_estimator(model_name, params)

        y_prob = cross_val_predict(est, X, y, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = safe_auc(y, y_prob)
        f1 = float(f1_score(y, y_pred, zero_division=0))

        summary_rows.append({
            'Endpoint': endpoint,
            'FeatureSet': fs,
            'BestModel': model_name,
            'CV_AUC': round(auc, 4),
            'CV_F1_Event': round(f1, 4),
            'Positives': int(y.sum()),
            'N': int(len(y)),
            'Tuned_AUC': row['CV_AUC'],
            'BestParams': row['BestParams'],
        })

        pred_store[(endpoint, fs)] = {
            'y': y,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'X': X,
            'model_name': model_name,
            'params': params,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(['Endpoint', 'FeatureSet'])
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # ROC curves by endpoint (one curve per feature set with its best model).
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    for i, ep in enumerate(ENDPOINTS):
        ax = axes[i]
        sub = summary_df[summary_df['Endpoint'] == ep].sort_values('FeatureSet')
        for _, r in sub.iterrows():
            fs = r['FeatureSet']
            y = pred_store[(ep, fs)]['y']
            y_prob = pred_store[(ep, fs)]['y_prob']
            fpr, tpr, _ = roc_curve(y, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"{fs.split(':')[0]} | {r['BestModel']} | AUC={r['CV_AUC']:.3f}")

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(ep)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=8, loc='lower right')

    plt.suptitle('ROC Curves by Endpoint (best model per feature set)')
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=150, bbox_inches='tight')

    # Heatmaps for quick table-like comparison.
    auc_pivot = summary_df.pivot(index='FeatureSet', columns='Endpoint', values='CV_AUC')
    f1_pivot = summary_df.pivot(index='FeatureSet', columns='Endpoint', values='CV_F1_Event')

    fig_auc, ax_auc = plt.subplots(figsize=(9, 4.5))
    im1 = ax_auc.imshow(auc_pivot.values, vmin=0.45, vmax=0.85, cmap='RdYlGn', aspect='auto')
    ax_auc.set_xticks(range(len(auc_pivot.columns)))
    ax_auc.set_xticklabels(auc_pivot.columns)
    ax_auc.set_yticks(range(len(auc_pivot.index)))
    ax_auc.set_yticklabels(auc_pivot.index)
    for r in range(auc_pivot.shape[0]):
        for c in range(auc_pivot.shape[1]):
            val = auc_pivot.values[r, c]
            ax_auc.text(c, r, f"{val:.3f}", ha='center', va='center', fontsize=9, fontweight='bold')
    ax_auc.set_title('CV ROC-AUC (best model per feature set)')
    plt.colorbar(im1, ax=ax_auc, fraction=0.046)
    plt.tight_layout()
    plt.savefig(OUT_HEAT_AUC, dpi=150, bbox_inches='tight')

    fig_f1, ax_f1 = plt.subplots(figsize=(9, 4.5))
    im2 = ax_f1.imshow(f1_pivot.values, vmin=0.2, vmax=0.7, cmap='YlGnBu', aspect='auto')
    ax_f1.set_xticks(range(len(f1_pivot.columns)))
    ax_f1.set_xticklabels(f1_pivot.columns)
    ax_f1.set_yticks(range(len(f1_pivot.index)))
    ax_f1.set_yticklabels(f1_pivot.index)
    for r in range(f1_pivot.shape[0]):
        for c in range(f1_pivot.shape[1]):
            val = f1_pivot.values[r, c]
            ax_f1.text(c, r, f"{val:.3f}", ha='center', va='center', fontsize=9, fontweight='bold')
    ax_f1.set_title('CV F1(Event) (best model per feature set)')
    plt.colorbar(im2, ax=ax_f1, fraction=0.046)
    plt.tight_layout()
    plt.savefig(OUT_HEAT_F1, dpi=150, bbox_inches='tight')

    # Endpoint-best configuration for subgroup analysis and feature drivers.
    best_endpoint = summary_df.sort_values('CV_AUC', ascending=False).groupby('Endpoint', as_index=False).first()

    subgroup_rows = []
    driver_rows = []

    print('Running subgroup analysis and feature driver extraction...')
    for _, r in best_endpoint.iterrows():
        endpoint = r['Endpoint']
        fs = r['FeatureSet']
        y = pred_store[(endpoint, fs)]['y']
        y_prob = pred_store[(endpoint, fs)]['y_prob']
        y_pred = pred_store[(endpoint, fs)]['y_pred']
        X = pred_store[(endpoint, fs)]['X']
        model_name = pred_store[(endpoint, fs)]['model_name']
        params = pred_store[(endpoint, fs)]['params']

        # Subgroup metrics based on out-of-fold predictions of best endpoint model.
        for sg in SUBGROUP_VARS:
            for level in sorted(df[sg].dropna().unique()):
                idx = df[sg] == level
                y_s = y[idx]
                if y_s.empty:
                    continue
                n = int(idx.sum())
                pos = int(y_s.sum())
                neg = n - pos
                if pos < 5 or neg < 5:
                    subgroup_rows.append({
                        'Endpoint': endpoint,
                        'SubgroupVar': sg,
                        'Level': level,
                        'N': n,
                        'Positives': pos,
                        'CV_AUC': np.nan,
                        'CV_F1_Event': np.nan,
                        'Note': 'Too few positives/negatives',
                    })
                    continue

                auc_s = safe_auc(y_s, y_prob[idx])
                f1_s = float(f1_score(y_s, y_pred[idx], zero_division=0))
                subgroup_rows.append({
                    'Endpoint': endpoint,
                    'SubgroupVar': sg,
                    'Level': level,
                    'N': n,
                    'Positives': pos,
                    'CV_AUC': round(auc_s, 4),
                    'CV_F1_Event': round(f1_s, 4),
                    'Note': '',
                })

        # Fit best model on full dataset to inspect feature drivers.
        est = build_estimator(model_name, params)
        est.fit(X, y)
        imp_df = get_importances(est, model_name, X, y, list(X.columns))
        imp_df['Endpoint'] = endpoint
        imp_df['FeatureSet'] = fs
        imp_df['Model'] = model_name
        driver_rows.append(imp_df)

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(OUT_SUBGROUP, index=False)

    drivers_df = pd.concat(driver_rows, ignore_index=True)
    drivers_df.to_csv(OUT_DRIVERS, index=False)

    # Plot top 10 driver features for each endpoint-best model.
    fig_d, axes_d = plt.subplots(2, 2, figsize=(13, 10))
    axes_d = axes_d.ravel()
    for i, ep in enumerate(ENDPOINTS):
        ax = axes_d[i]
        sub = drivers_df[drivers_df['Endpoint'] == ep].head(10)
        ax.barh(sub['feature'][::-1], sub['importance'][::-1], color='teal')
        ax.set_title(ep)
        ax.set_xlabel('Importance (model-specific)')
    plt.suptitle('Top Feature Drivers (best model per endpoint)')
    plt.tight_layout()
    plt.savefig(OUT_DRIVERS_PLOT, dpi=150, bbox_inches='tight')

    # Imaging value summary: does imaging-only / mask add value vs clinical?
    imaging_rows = []
    for ep in ENDPOINTS:
        ep_sub = summary_df[summary_df['Endpoint'] == ep].set_index('FeatureSet')
        auc_a = ep_sub.loc['A: ROI stats', 'CV_AUC']
        auc_b = ep_sub.loc['B: ROI + mask', 'CV_AUC']
        auc_c = ep_sub.loc['C: ROI + clinical', 'CV_AUC']
        auc_d = ep_sub.loc['D: All combined', 'CV_AUC']
        imaging_rows.append({
            'Endpoint': ep,
            'A_ROI_only_AUC': round(float(auc_a), 4),
            'B_ROI_plus_mask_AUC': round(float(auc_b), 4),
            'C_ROI_plus_clinical_AUC': round(float(auc_c), 4),
            'D_all_features_AUC': round(float(auc_d), 4),
            'Delta_mask_over_ROI': round(float(auc_b - auc_a), 4),
            'Delta_clinical_over_ROI': round(float(auc_c - auc_a), 4),
            'Delta_all_over_clinical': round(float(auc_d - auc_c), 4),
        })

    imaging_df = pd.DataFrame(imaging_rows)
    imaging_df.to_csv(OUT_IMAGING, index=False)

    print('\nSaved outputs:')
    print(f'  - {OUT_SUMMARY}')
    print(f'  - {OUT_SUBGROUP}')
    print(f'  - {OUT_DRIVERS}')
    print(f'  - {OUT_IMAGING}')
    print(f'  - {OUT_ROC}')
    print(f'  - {OUT_HEAT_AUC}')
    print(f'  - {OUT_HEAT_F1}')
    print(f'  - {OUT_DRIVERS_PLOT}')

    print('\nBest endpoint-level configs (for subgroup analysis):')
    print(best_endpoint[['Endpoint', 'FeatureSet', 'BestModel', 'CV_AUC', 'CV_F1_Event']].to_string(index=False))

    print('\nImaging value deltas:')
    print(imaging_df.to_string(index=False))


if __name__ == '__main__':
    main()
