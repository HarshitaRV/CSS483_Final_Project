#!/usr/bin/env python3
"""Quick end-to-end demo on 20 sample patients.

Shows:
1) CSV previews (clinical + ML dataset)
2) Masking availability check in CT-Scan-SegOut
3) ML output on sample-only data

This is intended for class/demo use, not final scientific benchmarking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_sample_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    sample_ids_path = root / "tests" / "sample_inputs" / "ct_sample_patients_20.txt"
    clinical_csv = root / "data" / "processed" / "clinical_labels_clean.csv"
    dataset_csv = root / "data" / "processed" / "dataset_ml.csv"
    segout_dir = root / "CT-Scan-SegOut"

    sample_ids = load_sample_ids(sample_ids_path)

    print_section("DEMO STEP 1: SHOW CSV FILES")
    print(f"Sample ID list file: {sample_ids_path}")
    print(f"Number of sample patients: {len(sample_ids)}")
    print("First 5 sample IDs:", sample_ids[:5])

    clinical_df = pd.read_csv(clinical_csv)
    dataset_df = pd.read_csv(dataset_csv)

    print("\nclinical_labels_clean.csv preview:")
    print(clinical_df.head(5).to_string(index=False))

    print("\ndataset_ml.csv preview:")
    print(dataset_df.head(5).to_string(index=False))

    print_section("DEMO STEP 2: MASKING OUTPUT CHECK")
    found_masks: list[str] = []
    missing_masks: list[str] = []

    for pid in sample_ids:
        mask_path = segout_dir / f"{pid}.nrrd"
        if mask_path.exists():
            found_masks.append(pid)
        else:
            missing_masks.append(pid)

    print(f"Mask directory: {segout_dir}")
    print(f"Masks found: {len(found_masks)} / {len(sample_ids)}")
    if found_masks:
        print("Example found IDs:", found_masks[:10])
    if missing_masks:
        print("Missing IDs:", missing_masks)

    print_section("DEMO STEP 3: MACHINE LEARNING OUTPUT ON SAMPLE-20")

    sample_df = dataset_df[dataset_df["Patient_ID"].isin(sample_ids)].copy()
    sample_df = sample_df.dropna()

    print(f"Rows in sample ML dataset: {len(sample_df)}")
    if len(sample_df) < 10:
        raise RuntimeError("Not enough rows in sample set for demo training/testing.")

    label_counts = sample_df["Label"].value_counts().to_dict()
    print("Label distribution:", label_counts)

    feature_cols = [
        "roi_mean",
        "roi_std",
        "roi_p10",
        "roi_p90",
        "roi_median",
        "mask_pixels",
        "mask_roi_overlap",
    ]

    X = sample_df[feature_cols]
    y = sample_df["Label"].astype(int)

    stratify_y = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify_y,
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True)),
            ]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    for name, model in models.items():
        print("\n" + "-" * 40)
        print(name)
        print("-" * 40)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

        if len(set(y_test)) > 1 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC: {auc:.4f}")
        else:
            print("ROC-AUC: N/A (single class in test split or no probability output)")

    print("\n[DONE] Quick demo completed.")
    print("Note: On only 20 patients, metrics can vary due to tiny sample size.")


if __name__ == "__main__":
    main()
