"""
=========================================================
  PROJECT DEMO — Sarcopenia-Based Tumor Failure Prediction
  Run: python run_demo.py
=========================================================
"""

import os, subprocess, time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE         = "/Users/acchuaccount/CSS483"
OVERLAY      = os.path.join(BASE, "all_overlays/HN-CHUM-030_overlay.png")
DATASET      = os.path.join(BASE, "dataset_ml.csv")
AUC_CHART    = "/tmp/ml_auc_comparison.png"
FEAT_CHART   = "/tmp/rf_feature_importances.png"
FEATURE_COLS = ["roi_mean", "roi_std", "roi_p10", "roi_p90", "roi_median"]

def banner(step, title):
    print("\n" + "=" * 60)
    print(f"  STEP {step}: {title}")
    print("=" * 60)
    time.sleep(0.3)

def pause():
    input("\n  [Press Enter to continue...]")

# ── STEP 1: INPUT DATA ────────────────────────────────────
banner(1, "INPUT DATA — ~300 Head & Neck CT Scans")
df = pd.read_csv(DATASET)
print(f"\n  Patients     : {len(df)}")
print(f"  Institutions : CHUM, CHUS, HGJ, HMR")
print(f"  Features     : {FEATURE_COLS}")
print(f"  Labels       : {df['Label'].value_counts().to_dict()}  (0=no failure, 1=tumor failure)")
pause()

# ── STEP 2: DEEPSARCOPENIA OUTPUT ────────────────────────
banner(2, "DEEPSARCOPENIA — C3 Segmentation Output (pre-run)")
print("\n  DenseNet selected the C3 vertebra slice.")
print("  U-Net segmented the neck muscle region.")
print("  Opening overlay for patient HN-CHUM-030...")
subprocess.run(["open", OVERLAY])
time.sleep(1)
pause()

# ── STEP 3: FEATURE EXTRACTION ───────────────────────────
banner(3, "FEATURE EXTRACTION — Sarcopenia Biomarkers")
print("\n  From each mask: mean, std, p10, p90, median intensity.\n")
print(df[FEATURE_COLS].describe().round(2).to_string())
pause()

# ── STEP 4: SPLIT + TRAIN + EVALUATE ─────────────────────
banner(4, "TRAIN/TEST SPLIT + ML TRAINING")
X = df[FEATURE_COLS]
y = df["Label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\n  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples  (80/20 stratified)\n")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

to_train = [
    (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), "Logistic Regression"),
    (SVC(class_weight="balanced", probability=True, random_state=42),             "SVM"),
    (RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42), "Random Forest"),
]

results = {}
rf_model = None
for model, name in to_train:
    model.fit(X_train_sc, y_train)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = model.predict(X_test_sc)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = auc
    if name == "Random Forest":
        rf_model = model
    print(f"  {name:<22}  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["No failure", "Tumor failure"],
          zero_division=0, digits=3))
pause()

# ── STEP 5: CHARTS ───────────────────────────────────────
banner(5, "RESULTS — Opening ROC-AUC Chart + Feature Importances")
best  = max(results, key=results.get)
names = list(results.keys())
aucs  = list(results.values())

plt.figure(figsize=(7, 4))
bars = plt.bar(names, aucs, color=["#2E86AB", "#F18F01", "#2A9D8F"])
plt.ylim(0, 1.05)
plt.ylabel("ROC-AUC")
plt.title("Core Model Comparison — ROC-AUC (Test Set)")
for bar, val in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
             f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(AUC_CHART, dpi=150)
plt.close()

importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
plt.figure(figsize=(6, 3.5))
importances.plot(kind="barh", color="#2A9D8F")
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.savefig(FEAT_CHART, dpi=150)
plt.close()

subprocess.run(["open", AUC_CHART])
time.sleep(0.5)
subprocess.run(["open", FEAT_CHART])

print(f"\n  Best model       : {best}  (AUC = {results[best]:.4f})")
print(f"  Clinical takeaway: sarcopenia features predict tumor failure risk.")

# ── DONE ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO COMPLETE")
print("=" * 60)
print(f"""
  CT scans  →  C3 selection + segmentation (DeepSarcopenia)
            →  Feature extraction (mean/std/p10/p90/median)
            →  LR / SVM / Random Forest
            →  Best: {best}  (AUC = {results[best]:.4f})
            →  Predicts tumor failure risk
""")
