#Please edit the directories for all xlsx files and replace it with your own 
import numpy as np
import pandas as pd

clinical_xlsx = "/Users/acchuaccount/CSS483/data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx"
roi_csv = "/Users/acchuaccount/CSS483/data/processed/roi_features_overlay_only.csv"

out_clinical_csv = "/Users/acchuaccount/CSS483/data/processed/clinical_labels_all3_clean.csv"
out_freq_csv = "/Users/acchuaccount/CSS483/data/processed/clinical_frequency_comparison.csv"
out_merged_csv = "/Users/acchuaccount/CSS483/data/processed/dataset_ml_all3.csv"
out_merged_xlsx = "/Users/acchuaccount/CSS483/data/processed/final_dataset_all3.xlsx"

all_sheets = pd.read_excel(clinical_xlsx, sheet_name=None)
clin = pd.concat(all_sheets.values(), ignore_index=True)

clin = clin.rename(
    columns={
        "Patient #": "Patient_ID",
        "Locoregional": "Locoregional",
        "Death": "Death",
        "Distant": "Distant",
    }
)

required = ["Patient_ID", "Locoregional", "Death", "Distant"]
missing_cols = [c for c in required if c not in clin.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in clinical workbook: {missing_cols}")

for c in ["Locoregional", "Death", "Distant"]:
    clin[c] = pd.to_numeric(clin[c], errors="coerce")

clin = clin.dropna(subset=["Patient_ID"]).copy()
clin["Patient_ID"] = clin["Patient_ID"].astype(str).str.strip()

valid_mask = clin[["Locoregional", "Death", "Distant"]].isin([0, 1]).all(axis=1)
clin_valid = clin.loc[valid_mask, ["Patient_ID", "Locoregional", "Death", "Distant"]].copy()

clin_patient = (
    clin_valid.groupby("Patient_ID", as_index=False)[["Locoregional", "Death", "Distant"]].max()
)

# 1 only if all three events are present.
clin_patient["Label_all3"] = (
    (clin_patient["Locoregional"] == 1)
    & (clin_patient["Death"] == 1)
    & (clin_patient["Distant"] == 1)
).astype(int)

clin_patient.to_csv(out_clinical_csv, index=False)

n = len(clin_patient)
L = int((clin_patient["Locoregional"] == 1).sum())
Dth = int((clin_patient["Death"] == 1).sum())
Dst = int((clin_patient["Distant"] == 1).sum())
all3 = int((clin_patient["Label_all3"] == 1).sum())
any3 = int(((clin_patient[["Locoregional", "Death", "Distant"]].sum(axis=1)) >= 1).sum())
exactly2 = int(((clin_patient[["Locoregional", "Death", "Distant"]].sum(axis=1)) == 2).sum())

freq = pd.DataFrame(
    [
        {"Metric": "Total patients (valid 0/1 for all 3)", "Count": n, "Percent": 100.0},
        {"Metric": "Locoregional = 1", "Count": L, "Percent": (100.0 * L / n if n else np.nan)},
        {"Metric": "Death = 1", "Count": Dth, "Percent": (100.0 * Dth / n if n else np.nan)},
        {"Metric": "Distant = 1", "Count": Dst, "Percent": (100.0 * Dst / n if n else np.nan)},
        {"Metric": "Any of 3 = 1", "Count": any3, "Percent": (100.0 * any3 / n if n else np.nan)},
        {"Metric": "Exactly 2 of 3 = 1", "Count": exactly2, "Percent": (100.0 * exactly2 / n if n else np.nan)},
        {"Metric": "All 3 = 1 (Label_all3 = 1)", "Count": all3, "Percent": (100.0 * all3 / n if n else np.nan)},
    ]
)
freq["Percent"] = freq["Percent"].round(2)
freq.to_csv(out_freq_csv, index=False)

roi = pd.read_csv(roi_csv)
roi["Patient_ID"] = roi["Patient_ID"].astype(str).str.strip()
merged = pd.merge(roi, clin_patient, on="Patient_ID", how="inner")
merged.to_csv(out_merged_csv, index=False)

with pd.ExcelWriter(out_merged_xlsx, engine="openpyxl") as writer:
    merged.to_excel(writer, index=False, sheet_name="dataset_all3")
    freq.to_excel(writer, index=False, sheet_name="frequency_comparison")

print("clinical_rows_patient_level", len(clin_patient))
print("roi_rows", len(roi))
print("merged_rows", len(merged))
print("label_all3_pos", int((merged["Label_all3"] == 1).sum()))
print("label_all3_neg", int((merged["Label_all3"] == 0).sum()))
print("saved", out_clinical_csv)
print("saved", out_freq_csv)
print("saved", out_merged_csv)
print("saved", out_merged_xlsx)
print("--- frequency table ---")
print(freq.to_string(index=False))
