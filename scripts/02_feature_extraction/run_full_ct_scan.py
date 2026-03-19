import os
import pandas as pd
from src.infer_slice_selection import test_slice_selection
from src.infer_segmentation import test_segmentation

CT_PNG_ROOT = "/Users/acchuaccount/CSS483/CT-Scan"
CT_NRRD_ROOT = "/tmp/CT-Scan-NRRD"
SLICE_MODEL = "/Users/acchuaccount/CSS483/DeepSarcopenia/model/test/C3_Top_Selection_Model_Weight.hdf5"
SEG_MODEL = "/Users/acchuaccount/CSS483/DeepSarcopenia/model/test/C3_Top_Segmentation_Model_Weight.hdf5"
PRED_CSV = "/Users/acchuaccount/CSS483/DeepSarcopenia/data/RADCURE/quick_C3_pred.csv"
SEG_OUTPUT = "/tmp/CT-Scan-SegOut"

os.makedirs(CT_NRRD_ROOT, exist_ok=True)
os.makedirs(SEG_OUTPUT, exist_ok=True)

# 1. convert pngs -> nrrd
print("Converting PNG folders to NRRD...")
os.system(f". .venv/bin/activate && python convert_all_png_to_nrrd.py")

# 2. run slice selection
print("Running slice selection across all NRRD files ...")
test_slice_selection(CT_NRRD_ROOT, SLICE_MODEL, PRED_CSV)

# 3. run segmentation for selected slice(s)
print("Running segmentation using C3 outputs ...")
test_segmentation(
    img_dir=CT_NRRD_ROOT,
    model_weight_path=SEG_MODEL,
    slice_csv_path=PRED_CSV,
    output_dir=SEG_OUTPUT
)

# 4. summarize results
df = pd.read_csv(PRED_CSV)
print("Done. Pred CSV:", PRED_CSV)
print(df.head())
print("Seg outputs saved in", SEG_OUTPUT)