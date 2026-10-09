"""
train_models.py
===============
Software Defect Prediction - Main Training Pipeline
Datasets : PROMISE Repository (CK + OO + CC Metrics)
Models   : Random Forest, Decision Tree, SVM, XGBoost, LightGBM
Pipeline : StandardScaler + Engineered Features + SMOTE (training only)
"""

import argparse
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
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from data_utils import normalize_bug_labels

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDE = ["xalan"]

ALL_FEATURES = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc",
    "ca", "ce", "npm", "lcom3", "dam", "moa", "mfa",
    "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"
]

# ─────────────────────────────────────────────
# 1. LOAD & MERGE DATASETS
# ─────────────────────────────────────────────

def load_all_datasets():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    if not files:
        print("No CSV files found in '%s'." % DATASET_DIR)
        return None

    frames = []
    for f in files:
        name = os.path.basename(f).lower()
        if any(ex in name for ex in EXCLUDE):
            print("  SKIPPED %-30s | reason=excluded" % os.path.basename(f))
            continue
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
# 2. ENGINEERED FEATURES
# ─────────────────────────────────────────────

def add_engineered_features(X, feature_names):
    eps = 1e-6
    idx = {n: i for i, n in enumerate(feature_names)}

    cols = []
    names = []

    def safe(col):
        return X[:, idx[col]:idx[col]+1] if col in idx else np.zeros((len(X), 1))

    wmc = safe("wmc"); loc = safe("loc")
    cbo = safe("cbo"); rfc = safe("rfc"); lcom = safe("lcom")
    dit = safe("dit")

    cols += [np.log1p(wmc), np.log1p(loc), np.log1p(np.abs(lcom)),
             np.log1p(rfc), np.log1p(cbo), np.log1p(dit)]
    names += ["log_wmc", "log_loc", "log_lcom", "log_rfc", "log_cbo", "log_dit"]

    cols += [wmc / (loc + eps), cbo / (loc + eps),
             rfc / (wmc + eps), lcom / (wmc + eps),
             cbo + rfc, (wmc + rfc + cbo) / 3.0]
    names += ["wmc_per_loc", "cbo_per_loc", "rfc_per_wmc",
              "lcom_per_wmc", "coupling_sum", "complexity_avg"]

    eng = np.hstack(cols)
    return np.hstack([X, eng]), feature_names + names

# ─────────────────────────────────────────────
# 3. PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df):
    available = [c for c in ALL_FEATURES if c in df.columns]
    df = df[available + ["bug"]].copy()

    df["bug"] = normalize_bug_labels(df["bug"])
    df = df.dropna(subset=["bug"])
    df["bug"] = df["bug"].astype(int)

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()

    total     = len(df)
    defective = df["bug"].sum()
    clean     = total - defective
    print("\nDataset Statistics:")
    print("  Total Samples    : %d" % total)
    print("  Defective (1)    : %d  (%.1f%%)" % (defective, 100 * defective / total))
    print("  Clean (0)        : %d  (%.1f%%)" % (clean,     100 * clean     / total))
    return df, available

# ─────────────────────────────────────────────
# 4. THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────

def best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print("\n" + "-" * 50)
    print("  Model     : %s" % name)
    print("  Threshold : %.2f" % threshold)
    print("  Acc       : %.4f" % acc)
    print("  Prec      : %.4f" % prec)
    print("  Recall    : %.4f" % rec)
    print("  F1        : %.4f" % f1)
    print("  AUC       : %.4f" % auc)
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC-ROC": auc, "Threshold": threshold}

# ─────────────────────────────────────────────
# 6. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate(df, features):
    X = df[features].values
    y = df["bug"].values

    # Add engineered features
    X, features = add_engineered_features(X, features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    validation_scaler = StandardScaler()
    X_fit_sc = np.nan_to_num(
        validation_scaler.fit_transform(X_fit), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_val_sc = np.nan_to_num(
        validation_scaler.transform(X_val), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_fit_sm, y_fit_sm = SMOTE(random_state=42).fit_resample(X_fit_sc, y_fit)

    scaler = StandardScaler()
    X_train_sc = np.nan_to_num(
        scaler.fit_transform(X_train), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_test_sc = np.nan_to_num(
        scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(
        X_train_sc, y_train
    )

    u, c   = np.unique(y_train, return_counts=True)
    u2, c2 = np.unique(y_train_sm, return_counts=True)
    print("\nSMOTE Applied:")
    print("  Before — Clean: %d  Defective: %d" % (c[0], c[1]))
    print("  After  — Clean: %d  Defective: %d" % (c2[0], c2[1]))
    print("  Total features (raw + engineered): %d" % len(features))

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "SVM"          : SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "XGBoost"      : XGBClassifier(n_estimators=300, learning_rate=0.05,
                                        max_depth=6, eval_metric="logloss",
                                        random_state=42, n_jobs=-1),
        "LightGBM"     : lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                              num_leaves=63, random_state=42,
                                              n_jobs=-1, verbose=-1),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_fit_sm, y_fit_sm)
        thr = best_threshold(model, X_val_sc, y_val)
        model.fit(X_train_sm, y_train_sm)
        results.append(evaluate(name, model, X_test_sc, y_test, threshold=thr))
        trained[name] = model

    summary = pd.DataFrame(results).set_index("Model")
    print("\n\n===== SUMMARY (with SMOTE + Engineered Features) =====")
    print(summary.to_string())

    out_csv = os.path.join(OUTPUT_DIR, "results_summary.csv")
    summary.to_csv(out_csv)
    print("\nResults saved to %s" % out_csv)

    return trained, scaler, X_train_sm, X_test_sc, y_train_sm, y_test, features

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train software defect prediction models.")
    parser.add_argument(
        "--all-projects", action="store_true",
        help="Run the original pooled multi-project SMOTE evaluation instead of POI-specific tuning.",
    )
    args = parser.parse_args()

    if not args.all_projects:
        # Make the default command follow the active POI project and print the
        # actual held-out accuracy / precision / recall / F1 for each model.
        from train_poi_model import main as tune_poi_model

        tune_poi_model("f1", split_seed=2026)
        result_path = os.path.join(OUTPUT_DIR, "poi_f1_results_seed2026.csv")
        tuned = pd.read_csv(result_path)
        tuned["Family"] = tuned["Model"].str.split("__").str[0].str.split("_").str[0]
        best_by_family = tuned.loc[tuned.groupby("Family")["CV_F1"].idxmax()]
        show = best_by_family[best_by_family["Family"].isin(
            ["RandomForest", "DecisionTree", "SVM", "XGBoost", "LightGBM", "ExtraTrees", "HistGradientBoosting"]
        )].copy()
        show = show.set_index("Family").rename(index={"RandomForest": "Random Forest"})
        show = show.rename(columns={
            "Test_Accuracy": "Accuracy",
            "Test_Precision": "Precision",
            "Test_Recall": "Recall",
            "Test_F1": "F1",
            "Test_ROC_AUC": "AUC-ROC",
            "Threshold": "Threshold",
        })
        columns = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "Threshold"]
        print("\n===== POI-3.0 SUMMARY (fresh held-out split; selected by training CV F1) =====")
        print(show[columns].to_string(float_format=lambda value: f"{value:.4f}"))
        print("\nFull model comparison: %s" % result_path)
    else:
        print("=" * 60)
        print("  Software Defect Prediction - pooled PROMISE projects")
        print("=" * 60)

        df_raw = load_all_datasets()
        if df_raw is None:
            exit(1)

        df, features = preprocess(df_raw)
        train_and_evaluate(df, features)
