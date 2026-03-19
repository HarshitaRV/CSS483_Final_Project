import pandas as pd

file_path = "/Users/acchuaccount/CSS483/data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx"

all_sheets = pd.read_excel(file_path, sheet_name=None)
clinical_df = pd.concat(all_sheets.values(), ignore_index=True)

clinical_df = clinical_df.rename(columns={
    'Patient #': 'Patient_ID',
    'Locoregional': 'LR',
    'Distant': 'DM',
    'Time – diagnosis to last follow-up(days)': 'Followup_days'
})

def assign_label(row):
    if row['LR'] == 1 or row['DM'] == 1:
        return 1
    elif row['LR'] == 0 and row['DM'] == 0 and row['Followup_days'] >= 730:
        return 0
    else:
        return None

clinical_df['Label'] = clinical_df.apply(assign_label, axis=1)
clinical_df = clinical_df.dropna(subset=['Label'])

print("Positive:", sum(clinical_df['Label'] == 1))
print("Negative:", sum(clinical_df['Label'] == 0))

clinical_df[['Patient_ID', 'Label']].to_csv("/Users/acchuaccount/CSS483/data/processed/clinical_labels_clean.csv", index=False)

print("Saved /Users/acchuaccount/CSS483/data/processed/clinical_labels_clean.csv")