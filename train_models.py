"""
train_models.py
===============
Software Defect Prediction - Main Training Pipeline
Datasets : PROMISE Repository (CK Metrics)
Models   : Random Forest, Decision Tree, SVM, XGBoost
Pipeline : StandardScaler + SMOTE (training set only) — consistent with save_best_model.py
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

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
from imblearn.over_sampling import SMOTE

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# ─────────────────────────────────────────────
# 1. LOAD & MERGE DATASETS
# ─────────────────────────────────────────────

def load_all_datasets():
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not files:
        print("No CSV files found in '%s'." % DATASET_DIR)
        return None

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        label = next((c for c in df.columns if c in ["bug", "defect", "class", "label"]), None)
        if label is None:
            continue
        df = df.rename(columns={label: "bug"})
        frames.append(df)
        print("  Loaded %-30s | shape=%s" % (os.path.basename(f), df.shape))

    combined = pd.concat(frames, ignore_index=True)
    print("\nCombined dataset shape: %s" % str(combined.shape))
    return combined

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df):
    available = [c for c in CK_FEATURES if c in df.columns]
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

    # Dataset statistics
    total      = len(df)
    defective  = df["bug"].sum()
    clean      = total - defective
    print("\nDataset Statistics:")
    print("  Total Samples    : %d" % total)
    print("  Defective (1)    : %d  (%.1f%%)" % (defective, 100 * defective / total))
    print("  Clean (0)        : %d  (%.1f%%)" % (clean,     100 * clean     / total))
    return df, available

# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE  (with SMOTE — consistent pipeline)
# ─────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    print("\n" + "-" * 50)
    print("  Model : %s" % name)
    print("  Acc   : %.4f" % acc)
    print("  Prec  : %.4f" % prec)
    print("  Recall: %.4f" % rec)
    print("  F1    : %.4f" % f1)
    print("  AUC   : %.4f" % auc)
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC-ROC": auc}

def train_and_evaluate(df, features):
    X = df[features].values
    y = df["bug"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale — fit on training set only to prevent data leakage
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE — applied only to training set to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)

    u, c   = np.unique(y_train, return_counts=True)
    u2, c2 = np.unique(y_train_sm, return_counts=True)
    print("\nSMOTE Applied:")
    print("  Before — Clean: %d  Defective: %d" % (c[0], c[1]))
    print("  After  — Clean: %d  Defective: %d" % (c2[0], c2[1]))

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM"          : SVC(probability=True, random_state=42),
        "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)          # train on SMOTE-balanced data
        results.append(evaluate(name, model, X_test_sc, y_test))
        trained[name] = model

    summary = pd.DataFrame(results).set_index("Model")
    print("\n\n===== SUMMARY (with SMOTE) =====")
    print(summary.to_string())

    out_csv = os.path.join(OUTPUT_DIR, "results_summary.csv")
    summary.to_csv(out_csv)
    print("\nResults saved to %s" % out_csv)

    # Save test arrays for visualize.py
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_sc)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    print("X_test.npy and y_test.npy saved.")

    return trained, scaler, X_train_sm, X_test_sc, y_train_sm, y_test, features

# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Software Defect Prediction - PROMISE CK Metrics")
    print("=" * 60)

    df_raw = load_all_datasets()
    if df_raw is None:
        exit(1)

    df, features = preprocess(df_raw)
    train_and_evaluate(df, features)
