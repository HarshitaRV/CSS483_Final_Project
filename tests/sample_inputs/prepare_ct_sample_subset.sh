#!/usr/bin/env bash
set -euo pipefail

# Build a 20-patient CT sample subset for grading/tests.
# Default mode is symlink to avoid duplicating large data.
#
# Usage:
#   bash tests/sample_inputs/prepare_ct_sample_subset.sh
#   bash tests/sample_inputs/prepare_ct_sample_subset.sh --mode copy
#   bash tests/sample_inputs/prepare_ct_sample_subset.sh --mode symlink --dest tests/sample_inputs/CT-Scan-Sample20

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST_FILE="$ROOT_DIR/tests/sample_inputs/ct_sample_patients_20.txt"
SRC_DIR="$ROOT_DIR/CT-Scan"
DEST_DIR="$ROOT_DIR/tests/sample_inputs/CT-Scan-Sample20"
MODE="symlink"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --dest)
      DEST_DIR="$ROOT_DIR/$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "symlink" && "$MODE" != "copy" ]]; then
  echo "--mode must be 'symlink' or 'copy'"
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Missing patient list: $LIST_FILE"
  exit 1
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Missing source CT folder: $SRC_DIR"
  exit 1
fi

mkdir -p "$DEST_DIR"

while IFS= read -r patient || [[ -n "$patient" ]]; do
  [[ -z "$patient" ]] && continue
  src="$SRC_DIR/$patient"
  dst="$DEST_DIR/$patient"

  if [[ ! -d "$src" ]]; then
    echo "[WARN] Missing source patient folder: $src"
    continue
  fi

  if [[ -e "$dst" ]]; then
    rm -rf "$dst"
  fi

  if [[ "$MODE" == "symlink" ]]; then
    ln -s "$src" "$dst"
  else
    cp -R "$src" "$dst"
  fi

  echo "[OK] Added $patient"
done < "$LIST_FILE"

echo "[DONE] Created sample CT subset at: $DEST_DIR"
echo "[INFO] Mode: $MODE"
