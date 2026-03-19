import os
import pandas as pd

data_dir = "/Users/acchuaccount/Downloads/CT-Scan"
file_path = "/Users/acchuaccount/Downloads/INFOclinical_HN_Version2_30may2018.xlsx"

total_series = 0
total_images = 0

for folder in os.listdir(data_dir):

    folder_path = os.path.join(data_dir, folder)

    if os.path.isdir(folder_path):

        total_series += 1

        images = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        total_images += len(images)

# read all clinical sheets
all_sheets = pd.read_excel(file_path, sheet_name=None)

all_sheets.pop("Excluded", None)

clinical_df = pd.concat(all_sheets.values(), ignore_index=True)

print("Total Patients (CT folders):", total_series)
print("Total CT Images:", total_images)
print("Total Clinical Records:", len(clinical_df))
print("Sheets used:", list(all_sheets.keys()))