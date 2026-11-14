import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ----------------------------- CONFIG -----------------------------
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "landmarks.csv"
MODEL_DIR = ROOT / "models"
META_COLS = ["label", "filename", "split", "flipped"]
# ------------------------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)
angle_cols = [c for c in df.columns if c.startswith("angle_")]

if {"train", "test"}.issubset(set(df["split"].unique())):
    train_df = df[df["split"].isin(["train", "val"])]
    test_df = df[df["split"] == "test"]
    print("Using the dataset's own train/test split")
else:
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    print("No train/test folders found, using a random 80/20 split")

X_tr, y_tr = train_df.drop(columns=META_COLS), train_df["label"]
X_te, y_te = test_df.drop(columns=META_COLS), test_df["label"]
print(f"Train: {len(X_tr)} samples | Test: {len(X_te)} samples")

clf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_tr, y_tr)

pred = clf.predict(X_te)
print("\n", classification_report(y_te, pred))
print("Confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(
    confusion_matrix(y_te, pred, labels=clf.classes_),
    index=clf.classes_, columns=clf.classes_,
))

joblib.dump(clf, MODEL_DIR / "asana_classifier.joblib")

# Reference angles come from ORIGINAL (unflipped) training images only,
# so left/right are not blended together.
originals = train_df[train_df["flipped"] == 0]
ref = originals.groupby("label")[angle_cols].median()
tol = originals.groupby("label")[angle_cols].std()
with open(MODEL_DIR / "reference_angles.json", "w") as f:
    json.dump({"ref": ref.to_dict("index"), "tol": tol.to_dict("index")}, f, indent=2)

print(f"\nSaved to {MODEL_DIR}: asana_classifier.joblib, reference_angles.json")