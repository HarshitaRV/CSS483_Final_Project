# Sample Testing Inputs (20 CT Patients)

This folder provides a small, reproducible CT subset for grading/testing.

## Files

- `tests/sample_inputs/ct_sample_patients_20.txt`: fixed patient ID list (20 patients).
- `tests/sample_inputs/prepare_ct_sample_subset.sh`: creates a local sample CT folder from `CT-Scan/`.

## Build Sample Subset

From repository root:

```bash
bash tests/sample_inputs/prepare_ct_sample_subset.sh
```

Default behavior:

- Creates `tests/sample_inputs/CT-Scan-Sample20`
- Uses symlinks (fast, no data duplication)

To copy actual files (portable folder):

```bash
bash tests/sample_inputs/prepare_ct_sample_subset.sh --mode copy
```

## Notes for Grading

- If the grader has the original `CT-Scan/` folder, symlink mode is enough.
- If you need to share a portable sample folder, use `--mode copy` and package `tests/sample_inputs/CT-Scan-Sample20`.
