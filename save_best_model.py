# -*- coding: utf-8 -*-
"""
save_best_model.py
==================
Trains all models, selects the best by ROC-AUC score,
and saves it along with the fitted StandardScaler using joblib.

Outputs
-------
  best_model.pkl  - best performing classifier
  scaler.pkl      - fitted StandardScaler (must be used at inference time)

Usage
-----
  python save_best_model.py
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")

# CK metric features used across all datasets
CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# Output paths
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(SCRIPT_DIR, "scaler.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    """Load all CSV datasets, normalise columns, and return X, y, feature list."""
    frames = []
    for f in glob.glob(os.path.join(DATASET_DIR, "*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        # Detect the defect label column regardless of its name
        label = next((c for c in df.columns if c in ["bug", "defect", "class", "label"]), None)
        if label:
            df = df.rename(columns={label: "bug"})
            frames.append(df)

    combined  = pd.concat(frames, ignore_index=True)
    available = [c for c in CK_FEATURES if c in combined.columns]

    combined = combined[available + ["bug"]].copy()

    # Normalise defect label to binary int
    combined["bug"] = (
        combined["bug"].astype(str).str.lower().str.strip()
        .map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    )
    combined = combined.dropna(subset=["bug"])
    combined["bug"] = combined["bug"].astype(int)

    # Median imputation for missing numeric values
    for col in available:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].median())

    combined = combined.drop_duplicates()
    print("Dataset loaded  : %d samples, %d features" % (combined.shape[0], len(available)))
    print("Features used   :", available)
    print("Class balance   :", combined["bug"].value_counts().to_dict(), "\n")

    X = combined[available].values
    y = combined["bug"].values
    return X, y, available


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN ALL MODELS & SELECT BEST BY ROC-AUC
# ─────────────────────────────────────────────────────────────────────────────

def train_and_select_best(X, y):
    """
    Split data, apply SMOTE on training set only, train all models,
    evaluate on the held-out test set, and return the best model + scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on training data only - prevents data leakage
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Apply SMOTE only to the training set to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM"          : SVC(probability=True, random_state=42),
        "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
    }

    print("%-20s %10s" % ("Model", "ROC-AUC"))
    print("-" * 32)

    best_name  = None
    best_model = None
    best_auc   = -1.0

    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        y_prob    = model.predict_proba(X_test_sc)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        print("%-20s %10.4f" % (name, auc_score))

        if auc_score > best_auc:
            best_auc   = auc_score
            best_name  = name
            best_model = model

    print("-" * 32)
    print("\n[BEST] %s  (AUC = %.4f)\n" % (best_name, best_auc))
    return best_model, best_name, best_auc, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAVE MODEL & SCALER
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, model_name, auc_score):
    """Persist the best model and scaler to disk using joblib."""
    joblib.dump(model,  BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Saved: best_model.pkl ->", BEST_MODEL_PATH)
    print("Saved: scaler.pkl     ->", SCALER_PATH)
    print("\nModel info:")
    print("  Name    :", model_name)
    print("  AUC-ROC : %.4f" % auc_score)
    print("  Type    :", type(model).__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Software Defect Prediction - Save Best Model")
    print("=" * 60 + "\n")

    X, y, features = load_and_preprocess()
    best_model, best_name, best_auc, scaler = train_and_select_best(X, y)
    save_artifacts(best_model, scaler, best_name, best_auc)

    print("\nDone. Run predict.py to load and use the saved model.")
