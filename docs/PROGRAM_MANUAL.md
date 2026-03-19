# Program Manual

## 1. Environment

- Python: 3.8+
- Recommended: activate the DeepSarcopenia virtual environment when running DeepSarcopenia scripts.

```bash
cd /Users/acchuaccount/CSS483/DeepSarcopenia
source .venv/bin/activate
```

## 2. Data Locations

- Raw clinical workbook: `data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx`
- Main processed dataset: `data/processed/dataset_ml.csv`
- CT folders: `CT-Scan/`

## 3. Typical Workflow

### Step A: Build labels / prepare data

```bash
cd /Users/acchuaccount/CSS483
python scripts/01_data_preparation/prepare_dataset.py
```

### Step B: Run baseline ML

```bash
python scripts/03_model_training/train_ml_baseline.py
```

Expected outputs include printed classification metrics and ROC-AUC plus saved plots under `/tmp/`.

### Step C: Run advanced ML and visual summaries

```bash
python scripts/03_model_training/train_ml_advanced.py
python scripts/04_analysis_visualization/plot_modelwise_roc_heatmaps_subgroups.py
```

Expected outputs include advanced metrics CSV and ROC/heatmap artifacts under `/tmp/`.

## 4. DeepSarcopenia Inference Path (optional)

If you need to rerun C3 slice selection + segmentation pipeline:

```bash
python scripts/02_feature_extraction/run_full_ct_scan.py
```

This step depends on model weights and DeepSarcopenia environment setup.

## 5. Troubleshooting

- If file-not-found errors occur, confirm paths inside scripts and data placement under `data/`.
- If TensorFlow/model errors occur, verify DeepSarcopenia model weights are present.
- For permission issues in `/tmp`, switch output locations to writable local paths.

## 6. Testing Files for Grading

Run the provided smoke tests from repository root:

```bash
bash tests/run_smoke_tests.sh quick
```

For a full run (baseline + advanced + visualization):

```bash
bash tests/run_smoke_tests.sh full
```

Related files:

- `tests/TESTING_FILES.md`
- `tests/run_smoke_tests.sh`
- `tests/expected_artifacts.txt`
- `tests/SUBMISSION_CHECKLIST.md`
