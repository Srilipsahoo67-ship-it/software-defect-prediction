"""
per_project_model.py
====================
Train a separate tuned LightGBM model per PROMISE project.
Uses leave-one-out style: train on ALL other projects, test on target project.
Also trains a within-project model (80/20 split) for maximum accuracy.
Saves best per-project models and prints a full accuracy report.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from data_utils import normalize_bug_labels

SEED = 42
np.random.seed(SEED)

BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "datasets")
MODELS_DIR  = os.path.join(BASE, "per_project_models")
os.makedirs(MODELS_DIR, exist_ok=True)

EXCLUDE = ["xalan"]

ALL_FEATURES = [
    "wmc","dit","noc","cbo","rfc","lcom","loc",
    "ca","ce","npm","lcom3","dam","moa","mfa","cam","ic","cbm","amc",
    "max_cc","avg_cc"
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_project(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
    if not lbl:
        return None
    df = df.rename(columns={lbl: "bug"})
    df["bug"] = normalize_bug_labels(df["bug"])
    df = df.dropna(subset=["bug"])
    df["bug"] = df["bug"].astype(int)
    return df

def get_XY(df):
    avail = [c for c in ALL_FEATURES if c in df.columns]
    X = df[avail].copy()
    for col in avail:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    X = X.values.astype(float)
    y = df["bug"].values
    return X, y

def add_features(X):
    """Add engineered features: log transforms + ratios."""
    eps = 1e-6
    # indices in ALL_FEATURES: wmc=0,loc=6,cbo=3,rfc=4,lcom=5
    wmc  = X[:, 0:1];  loc  = X[:, 6:7]
    cbo  = X[:, 3:4];  rfc  = X[:, 4:5];  lcom = X[:, 5:6]
    eng = np.hstack([
        np.log1p(wmc), np.log1p(loc), np.log1p(lcom),
        np.log1p(rfc), np.log1p(cbo),
        wmc  / (loc  + eps),
        cbo  / (loc  + eps),
        rfc  / (wmc  + eps),
        lcom / (wmc  + eps),
        cbo  + rfc,
    ])
    return np.hstack([X, eng])

def build_stacking_model():
    base = [
        ("lgbm", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                     num_leaves=50, random_state=SEED,
                                     n_jobs=-1, verbose=-1)),
        ("xgb",  XGBClassifier(n_estimators=400, learning_rate=0.05,
                                max_depth=5, eval_metric="logloss",
                                random_state=SEED, n_jobs=-1)),
        ("rf",   RandomForestClassifier(n_estimators=300, random_state=SEED,
                                         n_jobs=-1)),
    ]
    meta = LogisticRegression(max_iter=1000, random_state=SEED)
    return StackingClassifier(estimators=base, final_estimator=meta,
                               cv=5, n_jobs=-1, passthrough=False)

def best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ── load all projects ─────────────────────────────────────────────────────────

projects = {}
for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
    name = os.path.basename(f).replace(".csv", "")
    if any(ex in name.lower() for ex in EXCLUDE):
        print(f"  SKIP {name}")
        continue
    df = load_project(f)
    if df is not None:
        projects[name] = df
        print(f"  Loaded {name:20s}  rows={len(df):4d}  defect_rate={df['bug'].mean():.1%}")

print(f"\nTotal projects: {len(projects)}\n")

# ── per-project within-split training ────────────────────────────────────────

print("=" * 65)
print("  WITHIN-PROJECT MODELS  (80% train / 20% test per project)")
print("=" * 65)

results = []

for proj_name, df in projects.items():
    X, y = get_XY(df)
    X = add_features(X)

    # need at least 2 classes and enough samples
    if len(np.unique(y)) < 2 or len(y) < 30:
        print(f"  {proj_name}: skipped (too few samples or single class)")
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    X_tr_sc  = np.nan_to_num(X_tr_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_sc  = np.nan_to_num(X_te_sc,  nan=0.0, posinf=0.0, neginf=0.0)

    # balance only if minority class has enough samples
    minority = min(np.bincount(y_tr))
    if minority >= 6:
        k = min(5, minority - 1)
        sm = SMOTE(random_state=SEED, k_neighbors=k)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_sc, y_tr)
    else:
        X_tr_bal, y_tr_bal = X_tr_sc, y_tr

    # stacking ensemble
    model = build_stacking_model()
    model.fit(X_tr_bal, y_tr_bal)

    # threshold tuning on a small validation split from training data
    X_t2, X_val, y_t2, y_val = train_test_split(
        X_tr_bal, y_tr_bal, test_size=0.15, random_state=SEED, stratify=y_tr_bal
    )
    model2 = build_stacking_model()
    model2.fit(X_t2, y_t2)
    thr = best_threshold(model2, X_val, y_val)

    # final evaluation
    probs = model.predict_proba(X_te_sc)[:, 1]
    y_pred = (probs >= thr).astype(int)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.0
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)

    results.append({
        "Project"  : proj_name,
        "Samples"  : len(df),
        "DefRate"  : f"{y.mean():.1%}",
        "Accuracy" : round(acc, 4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec, 4),
        "F1"       : round(f1, 4),
        "AUC-ROC"  : round(auc, 4),
        "Threshold": round(thr, 2),
    })

    print(f"\n  {proj_name}")
    print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  thr={thr:.2f}")
    print(f"    {classification_report(y_te, y_pred, target_names=['Clean','Defective'], zero_division=0)}")

    # save model + scaler
    joblib.dump({"model": model, "scaler": scaler, "threshold": thr},
                os.path.join(MODELS_DIR, f"{proj_name}_model.pkl"))

# ── summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

avg_acc = res_df["Accuracy"].mean()
max_acc = res_df["Accuracy"].max()
min_acc = res_df["Accuracy"].min()

print(f"\n  Average Accuracy : {avg_acc:.4f}  ({avg_acc*100:.2f}%)")
print(f"  Best    Accuracy : {max_acc:.4f}  ({max_acc*100:.2f}%)")
print(f"  Worst   Accuracy : {min_acc:.4f}  ({min_acc*100:.2f}%)")

above_90 = (res_df["Accuracy"] >= 0.90).sum()
above_85 = (res_df["Accuracy"] >= 0.85).sum()
print(f"\n  Projects >= 90% accuracy : {above_90}/{len(res_df)}")
print(f"  Projects >= 85% accuracy : {above_85}/{len(res_df)}")

res_df.to_csv(os.path.join(BASE, "results", "per_project_results.csv"), index=False)
print(f"\n  Results saved -> results/per_project_results.csv")
print(f"  Models  saved -> per_project_models/")
