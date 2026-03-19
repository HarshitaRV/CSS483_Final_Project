#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test runner for the reorganized CSS483 project.
# Usage:
#   bash tests/run_smoke_tests.sh quick
#   bash tests/run_smoke_tests.sh full
# Optional:
#   PYTHON_BIN=/path/to/python bash tests/run_smoke_tests.sh quick

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-quick}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/DeepSarcopenia/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/DeepSarcopenia/.venv/bin/python"
else
  PY="python3"
fi

echo "[INFO] Project root: $ROOT_DIR"
echo "[INFO] Python: $PY"
echo "[INFO] Mode: $MODE"

run_cmd() {
  echo "[RUN] $*"
  "$@"
}

check_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f"
    exit 1
  fi
  echo "[OK] Found: $f"
}

check_artifact() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[WARN] Expected artifact not found: $f"
  else
    echo "[OK] Artifact: $f"
  fi
}

# Core required inputs
check_file "data/processed/dataset_ml.csv"
check_file "data/processed/clinical_labels_clean.csv"

# Stage 1: baseline model
run_cmd "$PY" "scripts/03_model_training/train_ml_baseline.py"

if [[ "$MODE" == "full" ]]; then
  # Stage 2: advanced model + plotting
  run_cmd "$PY" "scripts/03_model_training/train_ml_advanced.py"
  run_cmd "$PY" "scripts/04_analysis_visualization/plot_modelwise_roc_heatmaps_subgroups.py"
fi

# Stage 3: expected output checks
check_artifact "/tmp/ml_auc_comparison.png"
check_artifact "/tmp/rf_feature_importances.png"
check_artifact "/tmp/ml_advanced_results.csv"
check_artifact "/tmp/roc_panel_models_by_endpoint.png"
check_artifact "/tmp/heatmap_auc_by_model.png"
check_artifact "/tmp/heatmap_f1_by_model.png"

echo "[DONE] Smoke tests finished."
