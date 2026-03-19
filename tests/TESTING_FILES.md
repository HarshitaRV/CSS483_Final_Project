# Testing Files and Validation Guide

This folder now contains runnable testing files (not only instructions):

- `tests/run_smoke_tests.sh`: end-to-end smoke test runner.
- `tests/expected_artifacts.txt`: expected output artifact paths.
- `tests/TESTING_FILES.md`: this guide.

## Quick Smoke Test

From repository root:

```bash
bash tests/run_smoke_tests.sh quick
```

What it does:

1. Verifies key input files exist.
2. Runs baseline model training script.
3. Checks expected output artifacts in `/tmp`.

## Full Smoke Test

```bash
bash tests/run_smoke_tests.sh full
```

What it does:

1. Runs everything in quick mode.
2. Runs advanced model training.
3. Runs ROC/heatmap visualization script.
4. Re-checks all expected output artifacts.

## Python Environment Notes

- The smoke script will automatically use `DeepSarcopenia/.venv/bin/python` when available.
- You can override with your own interpreter:

```bash
PYTHON_BIN=/path/to/python bash tests/run_smoke_tests.sh quick
```

## Expected Artifacts

See `tests/expected_artifacts.txt`, which currently includes:

- `/tmp/ml_auc_comparison.png`
- `/tmp/rf_feature_importances.png`
- `/tmp/ml_advanced_results.csv`
- `/tmp/roc_panel_models_by_endpoint.png`
- `/tmp/heatmap_auc_by_model.png`
- `/tmp/heatmap_f1_by_model.png`

## Suggested Submission Attachments

- Private GitHub repository link with collaborator added.
- `README.md`
- `docs/PROGRAM_MANUAL.md`
- `tests/TESTING_FILES.md`
- `tests/run_smoke_tests.sh`
- `tests/expected_artifacts.txt`
- Key generated figures/CSVs used in report.
