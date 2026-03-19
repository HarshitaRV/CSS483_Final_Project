import pandas as pd

summary = pd.read_csv('/tmp/ml_auc_f1_by_endpoint_featureset.csv')
sub = pd.read_csv('/tmp/ml_subgroup_results.csv')
drv = pd.read_csv('/tmp/ml_feature_drivers.csv')
img = pd.read_csv('/tmp/ml_imaging_value_summary.csv')

print('=== AUC/F1 by Endpoint + FeatureSet (best model per set) ===')
print(summary[['Endpoint','FeatureSet','BestModel','CV_AUC','CV_F1_Event']].sort_values(['Endpoint','FeatureSet']).to_string(index=False))

print('\n=== Subgroup AUC range by variable (higher spread = stronger heterogeneity) ===')
clean = sub.dropna(subset=['CV_AUC']).copy()
rows = []
for (ep, var), g in clean.groupby(['Endpoint','SubgroupVar']):
    g = g.sort_values('CV_AUC')
    lo = g.iloc[0]
    hi = g.iloc[-1]
    rows.append({
        'Endpoint': ep,
        'SubgroupVar': var,
        'AUC_min': round(lo['CV_AUC'], 4),
        'Level_min': lo['Level'],
        'AUC_max': round(hi['CV_AUC'], 4),
        'Level_max': hi['Level'],
        'Spread': round(hi['CV_AUC'] - lo['CV_AUC'], 4),
    })
spread = pd.DataFrame(rows).sort_values(['Endpoint','Spread'], ascending=[True, False])
print(spread.to_string(index=False))

print('\n=== Top 5 feature drivers per endpoint ===')
for ep in sorted(drv['Endpoint'].unique()):
    top = drv[drv['Endpoint'] == ep].head(5)
    print(f'\n[{ep}]')
    print(top[['feature','importance']].to_string(index=False))

print('\n=== Imaging value summary ===')
print(img.to_string(index=False))
