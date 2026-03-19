import pandas as pd

xl = pd.ExcelFile('/Users/acchuaccount/CSS483/data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx')
sheets = []
for s in ['HGJ', 'CHUS', 'HMR', 'CHUM']:
    df = xl.parse(s)
    df['_sheet'] = s
    sheets.append(df)
all_clin = pd.concat(sheets, ignore_index=True)
all_clin.rename(columns={'Patient #': 'Patient_ID'}, inplace=True)

print('N-stage:', all_clin['N-stage'].value_counts(dropna=False).to_dict())
print('Age nulls:', all_clin['Age'].isna().sum())
print('Age range:', all_clin['Age'].min(), '-', all_clin['Age'].max())

ml = pd.read_csv('/Users/acchuaccount/CSS483/data/processed/dataset_ml.csv')
merged = ml.merge(
    all_clin[['Patient_ID', 'Age', 'Sex', 'Primary Site', 'T-stage', 'N-stage',
              'HPV status', 'Locoregional', 'Distant', 'Death']],
    on='Patient_ID', how='left'
)
print('merged rows:', len(merged))
print('HPV nulls after merge:', merged['HPV status'].isna().sum(), 'of', len(merged))
print('Age nulls after merge:', merged['Age'].isna().sum())
print('clin_merge nulls:', merged[['Age', 'Sex', 'Primary Site']].isna().sum().to_dict())
print('Locoregional 1s:', merged['Locoregional'].sum(), '  Distant 1s:', merged['Distant'].sum(), '  Death 1s:', merged['Death'].sum())
print('HPV value_counts:', merged['HPV status'].value_counts(dropna=False).to_dict())
print('N-stage counts:', merged['N-stage'].value_counts(dropna=False).to_dict())
print('T-stage counts:', merged['T-stage'].value_counts(dropna=False).to_dict())
