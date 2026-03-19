import pandas as pd

m = pd.read_csv('/tmp/model_feature_endpoint_cv_metrics.csv')

# Best feature set for each endpoint/model (used in ROC panel).
best_em = m.sort_values('CV_AUC', ascending=False).groupby(['Endpoint','Model'], as_index=False).first()
print('=== ROC panel mapping (endpoint x model best feature set) ===')
print(best_em[['Endpoint','Model','FeatureSet','CV_AUC','CV_F1_Event']].sort_values(['Endpoint','Model']).to_string(index=False))

# Overall best per endpoint.
best_ep = m.sort_values('CV_AUC', ascending=False).groupby('Endpoint', as_index=False).first()
print('\n=== Overall best per endpoint ===')
print(best_ep[['Endpoint','Model','FeatureSet','CV_AUC','CV_F1_Event']].to_string(index=False))

# AUC spread by subgroup from prior subgroup CSV.
sub = pd.read_csv('/tmp/ml_subgroup_results.csv')
sub = sub.dropna(subset=['CV_AUC'])
sub = sub[sub['SubgroupVar'].isin(['N_group','Sex_group','HPV_group','sarcopenia_proxy'])]
rows = []
for (ep, var), g in sub.groupby(['Endpoint','SubgroupVar']):
    rows.append({
        'Endpoint': ep,
        'Subgroup': var,
        'AUC_min': round(g['CV_AUC'].min(), 4),
        'AUC_max': round(g['CV_AUC'].max(), 4),
        'Spread': round(g['CV_AUC'].max() - g['CV_AUC'].min(), 4),
    })
out = pd.DataFrame(rows).sort_values(['Endpoint','Spread'], ascending=[True, False])
print('\n=== Subgroup heterogeneity (AUC spread) ===')
print(out.to_string(index=False))
