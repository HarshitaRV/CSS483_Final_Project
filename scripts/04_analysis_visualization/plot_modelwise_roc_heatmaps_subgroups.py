"""Generate model-wise ROC curves, AUC/F1 heatmaps, and subgroup AUC plots."""

import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

warnings.filterwarnings('ignore')

RESULTS_CSV = '/tmp/ml_advanced_results.csv'

OUT_METRICS = '/tmp/model_feature_endpoint_cv_metrics.csv'
OUT_PANEL_ROC = '/tmp/roc_panel_models_by_endpoint.png'
OUT_HEATMAP_AUC = '/tmp/heatmap_auc_by_model.png'
OUT_HEATMAP_F1 = '/tmp/heatmap_f1_by_model.png'
OUT_SUBGROUP_PREFIX = '/tmp/subgroup_auc_'

ENDPOINT_ORDER = ['Death', 'Distant', 'Locoregional', 'LR_or_DM']
FEATURE_ORDER = ['A: ROI stats', 'B: ROI + mask', 'C: ROI + clinical', 'D: All combined']
MODEL_ORDER = ['LogReg', 'SVM', 'RandomForest']


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

    # Exploratory sarcopenia proxy from mask area.
    q1 = df['mask_pixels'].quantile(0.25)
    df['sarcopenia_proxy'] = np.where(df['mask_pixels'] <= q1, 'Low area (Q1)', 'Higher area (Q2-4)')

    # Requested subgroup variables.
    df['N_stage_group'] = np.where(df['N_num'] >= 2, 'N2-3', 'N0-1')
    df['Sex_group'] = np.where(df['Sex'] == 'F', 'F', 'M')
    df['HPV_group'] = np.where(df['HPV status'] == '+', 'HPV+', np.where(df['HPV status'] == '-', 'HPV-', 'Unknown'))

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
    'Death': 'Death',
    'Distant': 'Distant',
    'Locoregional': 'Locoregional',
    'LR_or_DM': 'Label',
}

MODEL_COLORS = {
    'LogReg': '#1f77b4',
    'SVM': '#ff7f0e',
    'RandomForest': '#2ca02c',
}


def build_estimator(model_name, params):
    if model_name == 'LogReg':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=params.get('clf__C', 1.0),
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
            ))
        ])

    if model_name == 'SVM':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                C=params.get('clf__C', 1.0),
                kernel=params.get('clf__kernel', 'rbf'),
                class_weight='balanced',
                probability=True,
                random_state=42,
            ))
        ])

    return RandomForestClassifier(
        max_depth=params.get('max_depth', 6),
        min_samples_leaf=params.get('min_samples_leaf', 5),
        n_estimators=params.get('n_estimators', 100),
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )


def safe_auc(y_true, y_score):
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def main():
    df = load_data()
    tuned = pd.read_csv(RESULTS_CSV)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

    # Full endpoint x feature set x model metrics using tuned params.
    records = []
    pred_store = {}

    for endpoint in ENDPOINT_ORDER:
        y = df[ENDPOINTS[endpoint]].fillna(0).astype(int)

        for fs in FEATURE_ORDER:
            X = df[FEAT_SETS[fs]].copy()
            for c in X.columns:
                if X[c].isna().any():
                    X[c] = X[c].fillna(X[c].median())

            for model_name in MODEL_ORDER:
                row = tuned[(tuned['Endpoint'] == endpoint) &
                            (tuned['FeatureSet'] == fs) &
                            (tuned['Model'] == model_name)]
                if row.empty:
                    continue

                params = ast.literal_eval(row.iloc[0]['BestParams'])
                est = build_estimator(model_name, params)

                y_prob = cross_val_predict(est, X, y, cv=cv, method='predict_proba')[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                auc = safe_auc(y, y_prob)
                f1 = float(f1_score(y, y_pred, zero_division=0))

                records.append({
                    'Endpoint': endpoint,
                    'FeatureSet': fs,
                    'Model': model_name,
                    'CV_AUC': round(auc, 4),
                    'CV_F1_Event': round(f1, 4),
                    'BestParams': row.iloc[0]['BestParams'],
                })

                pred_store[(endpoint, fs, model_name)] = {
                    'y': y,
                    'y_prob': y_prob,
                    'y_pred': y_pred,
                }

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(OUT_METRICS, index=False)

    # Step 1: ROC panel (one subplot per endpoint; one line per model).
    # For each model, use its best feature set for that endpoint.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    best_model_endpoint = (
        metrics_df.sort_values('CV_AUC', ascending=False)
        .groupby(['Endpoint', 'Model'], as_index=False)
        .first()
    )

    for i, endpoint in enumerate(ENDPOINT_ORDER):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        for model_name in MODEL_ORDER:
            sub = best_model_endpoint[
                (best_model_endpoint['Endpoint'] == endpoint) &
                (best_model_endpoint['Model'] == model_name)
            ]
            if sub.empty:
                continue

            fs = sub.iloc[0]['FeatureSet']
            auc = sub.iloc[0]['CV_AUC']
            y = pred_store[(endpoint, fs, model_name)]['y']
            y_prob = pred_store[(endpoint, fs, model_name)]['y_prob']
            fpr, tpr, _ = roc_curve(y, y_prob)

            ax.plot(
                fpr, tpr,
                color=MODEL_COLORS[model_name],
                lw=2,
                label=f"{model_name} ({fs.split(':')[0]}) AUC={auc:.3f}"
            )

        ax.set_title(endpoint)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', fontsize=8)

    plt.suptitle('5-Fold CV ROC Curves by Endpoint (LR/SVM/RF, consistent model colors)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PANEL_ROC, dpi=160, bbox_inches='tight')

    # Step 2: heatmaps AUC/F1 with model-separated panels.
    # Axes requested: y=endpoints, x=feature sets.
    fig_auc, ax_auc = plt.subplots(1, 3, figsize=(16, 4.8))
    fig_f1, ax_f1 = plt.subplots(1, 3, figsize=(16, 4.8))

    for j, model_name in enumerate(MODEL_ORDER):
        sub = metrics_df[metrics_df['Model'] == model_name]
        piv_auc = sub.pivot(index='Endpoint', columns='FeatureSet', values='CV_AUC').reindex(
            index=ENDPOINT_ORDER, columns=FEATURE_ORDER
        )
        piv_f1 = sub.pivot(index='Endpoint', columns='FeatureSet', values='CV_F1_Event').reindex(
            index=ENDPOINT_ORDER, columns=FEATURE_ORDER
        )

        im_auc = ax_auc[j].imshow(piv_auc.values, vmin=0.40, vmax=0.85, cmap='RdYlGn', aspect='auto')
        ax_auc[j].set_title(f'{model_name} AUC')
        ax_auc[j].set_xticks(range(len(FEATURE_ORDER)))
        ax_auc[j].set_xticklabels(['A', 'B', 'C', 'D'])
        ax_auc[j].set_yticks(range(len(ENDPOINT_ORDER)))
        ax_auc[j].set_yticklabels(ENDPOINT_ORDER)
        for r in range(len(ENDPOINT_ORDER)):
            for c in range(len(FEATURE_ORDER)):
                v = piv_auc.values[r, c]
                ax_auc[j].text(c, r, f'{v:.3f}', ha='center', va='center', fontsize=8, fontweight='bold')

        im_f1 = ax_f1[j].imshow(piv_f1.values, vmin=0.00, vmax=0.70, cmap='YlGnBu', aspect='auto')
        ax_f1[j].set_title(f'{model_name} F1(Event)')
        ax_f1[j].set_xticks(range(len(FEATURE_ORDER)))
        ax_f1[j].set_xticklabels(['A', 'B', 'C', 'D'])
        ax_f1[j].set_yticks(range(len(ENDPOINT_ORDER)))
        ax_f1[j].set_yticklabels(ENDPOINT_ORDER)
        for r in range(len(ENDPOINT_ORDER)):
            for c in range(len(FEATURE_ORDER)):
                v = piv_f1.values[r, c]
                ax_f1[j].text(c, r, f'{v:.3f}', ha='center', va='center', fontsize=8, fontweight='bold')

    fig_auc.suptitle('AUC Heatmaps by Model (y=Endpoint, x=Feature Set A-D)')
    fig_auc.tight_layout()
    fig_auc.colorbar(im_auc, ax=ax_auc, fraction=0.02, pad=0.02)
    fig_auc.savefig(OUT_HEATMAP_AUC, dpi=160, bbox_inches='tight')

    fig_f1.suptitle('F1(Event) Heatmaps by Model (y=Endpoint, x=Feature Set A-D)')
    fig_f1.tight_layout()
    fig_f1.colorbar(im_f1, ax=ax_f1, fraction=0.02, pad=0.02)
    fig_f1.savefig(OUT_HEATMAP_F1, dpi=160, bbox_inches='tight')

    # Step 3: subgroup plots using each endpoint's overall best combo.
    endpoint_best = (
        metrics_df.sort_values('CV_AUC', ascending=False)
        .groupby('Endpoint', as_index=False)
        .first()
    )

    subgroup_vars = {
        'N_stage_group': ['N0-1', 'N2-3'],
        'Sex_group': ['F', 'M'],
        'HPV_group': ['HPV+', 'HPV-', 'Unknown'],
        'sarcopenia_proxy': ['Low area (Q1)', 'Higher area (Q2-4)'],
    }

    for sg_var, level_order in subgroup_vars.items():
        fig_sg, axes_sg = plt.subplots(2, 2, figsize=(13, 9))
        axes_sg = axes_sg.ravel()

        for i, endpoint in enumerate(ENDPOINT_ORDER):
            ax = axes_sg[i]
            row = endpoint_best[endpoint_best['Endpoint'] == endpoint].iloc[0]
            fs = row['FeatureSet']
            model = row['Model']
            y = pred_store[(endpoint, fs, model)]['y']
            y_prob = pred_store[(endpoint, fs, model)]['y_prob']
            y_pred = pred_store[(endpoint, fs, model)]['y_pred']

            auc_vals = []
            labels = []
            for lv in level_order:
                idx = (df[sg_var] == lv)
                y_s = y[idx]
                if len(y_s) == 0 or pd.Series(y_s).nunique() < 2:
                    auc_vals.append(np.nan)
                else:
                    auc_vals.append(roc_auc_score(y_s, y_prob[idx]))
                labels.append(lv)

            bars = ax.bar(labels, auc_vals, color='#4c78a8')
            for b, v in zip(bars, auc_vals):
                if np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, 0.02, 'NA', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

            valid = [v for v in auc_vals if not np.isnan(v)]
            spread = (max(valid) - min(valid)) if len(valid) >= 2 else np.nan
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{endpoint} | {model} {fs.split(':')[0]} | spread={spread:.3f}" if not np.isnan(spread) else f"{endpoint} | {model} {fs.split(':')[0]}")
            ax.set_ylabel('AUC')

        fig_sg.suptitle(f'Subgroup AUC Heterogeneity by {sg_var}', fontsize=12)
        fig_sg.tight_layout()
        out_path = f"{OUT_SUBGROUP_PREFIX}{sg_var}.png"
        fig_sg.savefig(out_path, dpi=160, bbox_inches='tight')

    print('Saved:')
    print(f'  {OUT_METRICS}')
    print(f'  {OUT_PANEL_ROC}')
    print(f'  {OUT_HEATMAP_AUC}')
    print(f'  {OUT_HEATMAP_F1}')
    print(f'  {OUT_SUBGROUP_PREFIX}N_stage_group.png')
    print(f'  {OUT_SUBGROUP_PREFIX}Sex_group.png')
    print(f'  {OUT_SUBGROUP_PREFIX}HPV_group.png')
    print(f'  {OUT_SUBGROUP_PREFIX}sarcopenia_proxy.png')


if __name__ == '__main__':
    main()
