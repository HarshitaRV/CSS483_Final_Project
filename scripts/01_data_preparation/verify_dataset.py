import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Verify ---
df = pd.read_csv("/Users/acchuaccount/CSS483/data/processed/dataset_ml.csv")
print("=== Step 1: Verify ===")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nNull counts:")
print(df.isnull().sum())
print("\nLabel distribution:")
print(df["Label"].value_counts())
print("Class balance: {:.1f}% positive".format(100 * (df["Label"] == 1.0).mean()))

# --- Step 2: Features + target ---
FEATURE_COLS = ["roi_mean", "roi_std", "roi_p10", "roi_p90", "roi_median"]
X = df[FEATURE_COLS]
y = df["Label"].astype(int)
print("\n=== Step 2: Features + Target ===")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Feature cols:", FEATURE_COLS)

# --- Step 3: Train/test split (stratified 80/20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\n=== Step 3: Train/Test Split ===")
print("X_train:", X_train.shape, "  X_test:", X_test.shape)
print("y_train distribution:", y_train.value_counts().to_dict())
print("y_test  distribution:", y_test.value_counts().to_dict())
print("Train positive rate: {:.1f}%".format(100 * y_train.mean()))
print("Test  positive rate: {:.1f}%".format(100 * y_test.mean()))
print("\nAll checks passed. Ready for ML training.")
