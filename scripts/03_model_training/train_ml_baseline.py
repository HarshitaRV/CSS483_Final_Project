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

# --- Data ---
df = pd.read_csv("/Users/acchuaccount/CSS483/data/processed/dataset_ml.csv")
FEATURE_COLS = ["roi_mean", "roi_std", "roi_p10", "roi_p90", "roi_median"]
X = df[FEATURE_COLS]
y = df["Label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale for LR + SVM (RF doesn't need it but doesn't hurt)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --- Models ---
lr  = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
svm = SVC(class_weight="balanced", probability=True, random_state=42)
rf  = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)

lr.fit(X_train_sc, y_train)
svm.fit(X_train_sc, y_train)
rf.fit(X_train_sc, y_train)

# --- Evaluate ---
def evaluate(model, name, X_eval):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No failure (0)", "Tumor failure (1)"]))
    print(f"ROC-AUC: {auc:.4f}")
    return auc

auc_lr  = evaluate(lr,  "Logistic Regression", X_test_sc)
auc_svm = evaluate(svm, "SVM",                 X_test_sc)
auc_rf  = evaluate(rf,  "Random Forest",        X_test_sc)

# --- Summary bar chart ---
models = ["Logistic Regression", "SVM", "Random Forest"]
aucs   = [auc_lr, auc_svm, auc_rf]

plt.figure(figsize=(7, 4))
bars = plt.bar(models, aucs, color=["steelblue", "darkorange", "seagreen"])
plt.ylim(0, 1.05)
plt.ylabel("ROC-AUC")
plt.title("Model Comparison — ROC-AUC (Test Set)")
for bar, val in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("/tmp/ml_auc_comparison.png", dpi=120)
plt.close()
print("\nAUC chart saved: /tmp/ml_auc_comparison.png")

# --- RF feature importances ---
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
plt.figure(figsize=(6, 3.5))
importances.plot(kind="barh", color="seagreen")
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.savefig("/tmp/rf_feature_importances.png", dpi=120)
plt.close()
print("Feature importance chart saved: /tmp/rf_feature_importances.png")
