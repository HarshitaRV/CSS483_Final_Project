# CSS483 Project Organization

This repository has been reorganized to separate code, data, and documentation.

## Folder Structure

```text
CSS483/
  scripts/
    01_data_preparation/
    02_feature_extraction/
    03_model_training/
    04_analysis_visualization/
    05_utilities/
  data/
    clinical_raw/
    processed/
  docs/
  tests/
  DeepSarcopenia/
  CT-Scan/
  CT-Scan-SegOut/
  all_overlays/
  presentation_assets/
```

## Python Script Catalog

Note: This catalog documents project-level scripts and key custom wrappers. The `DeepSarcopenia/src/` directory contains many upstream model internals.

### `scripts/01_data_preparation`

- `scripts/01_data_preparation/CSS483.py`: dataset inventory script; counts CT folders/PNG slices and summarizes merged clinical workbook records.
- `scripts/01_data_preparation/prepare_dataset.py`: creates main binary tumor-failure label (`Label`) from clinical workbook (`LR`/`DM` with follow-up rules).
- `scripts/01_data_preparation/inspect_clinical.py`: quick inspection helper for clinical fields and quality checks.
- `scripts/01_data_preparation/build_final_all3_dataset.py`: builds strict `all3` dataset/labels (`Locoregional`, `Death`, and `Distant` all positive).
- `scripts/01_data_preparation/verify_dataset.py`: verifies final ML dataset integrity and feature availability.
- `scripts/01_data_preparation/tmp_fold_counts.py`: utility script to inspect fold/class counts.

### `scripts/02_feature_extraction`

- `scripts/02_feature_extraction/extract_muscle_features.py`: computes simple PNG-derived muscle proxy features (`muscle_ratio`, `muscle_density`).
- `scripts/02_feature_extraction/run_full_ct_scan.py`: orchestrates end-to-end DeepSarcopenia inference flow for CT inputs.

### `scripts/03_model_training`

- `scripts/03_model_training/train_ml_baseline.py`: baseline ML run (80/20 split) with Logistic Regression, SVM, Random Forest and ROC-AUC outputs.
- `scripts/03_model_training/train_ml_advanced.py`: advanced evaluation (5-fold CV, hyperparameter tuning, endpoint-wise comparisons, feature-set ablations).

### `scripts/04_analysis_visualization`

- `scripts/04_analysis_visualization/plot_modelwise_roc_heatmaps_subgroups.py`: produces model-wise ROC panels, AUC/F1 heatmaps, subgroup comparisons.
- `scripts/04_analysis_visualization/analyze_subgroups_and_roc.py`: subgroup ROC analysis utilities.
- `scripts/04_analysis_visualization/create_fig3_overlay_panel.py`: builds publication-style overlay panel figure from selected overlays.
- `scripts/04_analysis_visualization/create_paper_artifacts.py`: packages plot/metric artifacts for paper or report usage.
- `scripts/04_analysis_visualization/summarize_exploratory_outputs.py`: summarizes exploratory analysis outputs.
- `scripts/04_analysis_visualization/summarize_modelwise_plots.py`: summarizes generated model-wise visual outputs.

### `scripts/05_utilities`

- `scripts/05_utilities/run_demo.py`: quick demonstration script for pipeline/model behavior.
- `scripts/05_utilities/convert_and_test.py`: utility converter/test helper for local data processing.

### Key `DeepSarcopenia` Scripts Used

- `DeepSarcopenia/main.py`: upstream entrypoint for slice selection + preprocessing + segmentation steps.
- `DeepSarcopenia/src/infer_slice_selection.py`: DenseNet-based C3 slice prediction logic.
- `DeepSarcopenia/src/infer_segmentation.py`: U-Net-based C3 muscle mask inference.
- `DeepSarcopenia/src/slice_area_density.py`: CSA/density computation helpers from predicted masks.
- `DeepSarcopenia/get_CSA.py`: generates C3 area/density CSV outputs from predicted slices/masks.
- `DeepSarcopenia/clinical.py`: computes derived clinical sarcopenia outputs (including SMI path) when required metadata is available.
- `DeepSarcopenia/convert_all_png_to_nrrd.py`: converts PNG stack folders to temporary NRRD volumes for model input.
- `DeepSarcopenia/run_full_ct_scan.py`: wrapper script to run conversion, slice selection, segmentation, and summary output.
- `DeepSarcopenia/extract_csa_merge_clinical.py`: builds ROI/mask-derived feature table and merges with clinical labels.
- `DeepSarcopenia/generate_all_overlays.py`: generates QC overlays and ROI feature exports.

## CSV/XLSX Data Catalog

### `data/clinical_raw`

- `data/clinical_raw/INFOclinical_HN_Version2_30may2018.xlsx`: original TCIA clinical workbook used for labels and clinical covariates.

### `data/processed`

- `data/processed/clinical_labels_clean.csv`: main patient-level binary label table (`Patient_ID`, `Label`).
- `data/processed/clinical_labels_all3_clean.csv`: strict all-three-events clinical labels.
- `data/processed/clinical_frequency_comparison.csv`: frequency summary table for clinical endpoint combinations.
- `data/processed/roi_features.csv`: ROI/mask-derived feature table merged from extraction pipeline.
- `data/processed/roi_features_overlay_only.csv`: ROI feature table used for overlay/all3 workflows.
- `data/processed/sarcopenia_features.csv`: derived sarcopenia-related feature output table.
- `data/processed/dataset_ml.csv`: main supervised learning dataset (features + `Label`).
- `data/processed/dataset_ml_all3.csv`: strict all3 variant ML dataset.
- `data/processed/patient_image_counts.csv`: per-patient CT image-count metadata.
- `data/processed/final_dataset_all3.xlsx`: spreadsheet export of all3 merged dataset.

## Quick Run Examples

From repository root:

```bash
python scripts/03_model_training/train_ml_baseline.py
python scripts/03_model_training/train_ml_advanced.py
python scripts/04_analysis_visualization/plot_modelwise_roc_heatmaps_subgroups.py
```

## Supplements (Required for Submission)



### 1) Testing files and program manual

- Program manual: `docs/PROGRAM_MANUAL.md`
- Testing guide: `tests/TESTING_FILES.md`
- Runnable smoke tests: `tests/run_smoke_tests.sh`
- Expected outputs list: `tests/expected_artifacts.txt`
- Submission checklist: `tests/SUBMISSION_CHECKLIST.md`

These documents describe setup, commands, expected outputs, and smoke-test checks.
