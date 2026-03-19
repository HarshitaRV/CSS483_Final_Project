import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('/Users/acchuaccount/CSS483/data/processed/dataset_ml.csv')
xl = pd.ExcelFile('/Users/acchuaccount/CSS483/data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx')
clin = pd.concat([xl.parse(s) for s in ['HGJ', 'CHUS', 'HMR', 'CHUM']], ignore_index=True)
clin = clin.rename(columns={'Patient #': 'Patient_ID'})

m = df.merge(clin[['Patient_ID', 'Locoregional', 'Distant', 'Death']], on='Patient_ID', how='left')
for c in ['Locoregional', 'Distant', 'Death']:
    m[c] = m[c].fillna(0).astype(int)

endpoints = {
    'LR_or_DM': 'Label',
    'Locoregional': 'Locoregional',
    'Distant': 'Distant',
    'Death': 'Death',
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
print('N =', len(m))

for ep, col in endpoints.items():
    y = m[col].astype(int).values
    print('\n' + ep + f' positives={y.sum()} negatives={(1 - y).sum()}')
    for i, (_, te) in enumerate(cv.split([[0]] * len(y), y), 1):
        ys = y[te]
        print(f'  fold{i}: n={len(te)} pos={ys.sum()} neg={(1 - ys).sum()}')
