"""
Software Defect Prediction - Main Pipeline
Datasets: PROMISE Repository (CK Metrics)
Models: Random Forest, Decision Tree, SVM, XGBoost
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# ─────────────────────────────────────────────
# 1. LOAD & MERGE DATASETS
# ─────────────────────────────────────────────

def load_all_datasets():
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not files:
        print(f"No CSV files found in '{DATASET_DIR}'.")
        print("Run download_datasets.py first.")
        return None

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        label = next((c for c in df.columns if c in ["bug", "defect", "class", "label"]), None)
        if label is None:
            print(f"  Skipping {os.path.basename(f)} — no defect label column.")
            continue
        df = df.rename(columns={label: "bug"})
        frames.append(df)
        print(f"  Loaded {os.path.basename(f):30s} | shape={df.shape}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset shape: {combined.shape}")
    return combined

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df):
    available = [c for c in CK_FEATURES if c in df.columns]
    missing   = [c for c in CK_FEATURES if c not in df.columns]
    if missing:
        print(f"  Warning: features not found and skipped: {missing}")

    df = df[available + ["bug"]].copy()

    # Convert defect label to binary 0/1
    df["bug"] = df["bug"].astype(str).str.lower().str.strip()
    df["bug"] = df["bug"].map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["bug"])
    df["bug"] = df["bug"].astype(int)

    # Median imputation for missing feature values
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()
    print(f"\nAfter preprocessing: {df.shape}")
    print(f"Class distribution:\n{df['bug'].value_counts()}")
    return df, available

# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE
# ─────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    print(f"\n{'-'*50}")
    print(f"  Model : {name}")
    print(f"  Acc   : {acc:.4f}")
    print(f"  Prec  : {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1    : {f1:.4f}")
    print(f"  AUC   : {auc:.4f}" if isinstance(auc, float) else f"  AUC   : {auc}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC-ROC": auc}

def train_and_evaluate(df, features):
    X = df[features].values
    y = df["bug"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM"          : SVC(probability=True, random_state=42),
        "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        trained[name] = model

    summary = pd.DataFrame(results).set_index("Model")
    print("\n\n===== SUMMARY =====")
    print(summary.to_string())

    out_csv = os.path.join(SCRIPT_DIR, "results_summary.csv")
    summary.to_csv(out_csv)
    print(f"\nResults saved to {out_csv}")

    # Save test arrays for visualize.py
    np.save(os.path.join(SCRIPT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SCRIPT_DIR, "y_test.npy"), y_test)
    print("X_test.npy and y_test.npy saved.")

    return trained, scaler, X_train, X_test, y_train, y_test, features

# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Software Defect Prediction — PROMISE CK Metrics")
    print("=" * 60)

    df_raw = load_all_datasets()
    if df_raw is None:
        exit(1)

    df, features = preprocess(df_raw)
    train_and_evaluate(df, features)
