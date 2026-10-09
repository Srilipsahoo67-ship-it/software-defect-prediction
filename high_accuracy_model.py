"""
high_accuracy_model.py
======================
Evaluates transfer-learning variants using:
1. Transfer learning: pre-train on ALL other projects
2. Fine-tune on target project (80/20 split)
3. Stacking ensemble: LightGBM + XGBoost + RF
4. Threshold optimization per project
5. Engineered features (log transforms + ratios)

Thresholds are tuned on a validation split from target training data; this
script does not guarantee a particular accuracy.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, classification_report)
from imblearn.over_sampling import SMOTE
from data_utils import normalize_bug_labels

SEED = 42
np.random.seed(SEED)

BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "datasets")
MODELS_DIR  = os.path.join(BASE, "high_acc_models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

EXCLUDE = ["xalan"]

ALL_FEATURES = [
    "wmc","dit","noc","cbo","rfc","lcom","loc",
    "ca","ce","npm","lcom3","dam","moa","mfa","cam","ic","cbm","amc",
    "max_cc","avg_cc"
]

# ── feature engineering ───────────────────────────────────────────────────────

def add_features(X):
    eps = 1e-6
    wmc = X[:, 0:1]; loc = X[:, 6:7]
    cbo = X[:, 3:4]; rfc = X[:, 4:5]; lcom = X[:, 5:6]
    dit = X[:, 1:2]; noc = X[:, 2:3]
    eng = np.hstack([
        np.log1p(wmc), np.log1p(loc), np.log1p(np.abs(lcom)),
        np.log1p(rfc), np.log1p(cbo), np.log1p(dit),
        wmc  / (loc  + eps),
        cbo  / (loc  + eps),
        rfc  / (wmc  + eps),
        lcom / (wmc  + eps),
        cbo  + rfc,
        wmc  * cbo,
        (wmc + rfc + cbo) / 3.0,
    ])
    return np.hstack([X, eng])

# ── data loading ──────────────────────────────────────────────────────────────

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

def smote_safe(X, y, seed=SEED):
    counts = np.bincount(y)
    minority = counts.min()
    if minority < 2:
        return X, y
    k = min(5, minority - 1)
    sm = SMOTE(random_state=seed, k_neighbors=k)
    return sm.fit_resample(X, y)

# ── model builders ────────────────────────────────────────────────────────────

def make_lgbm(n=500):
    return lgb.LGBMClassifier(
        n_estimators=n, learning_rate=0.03, num_leaves=63,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1
    )

def make_xgb(n=500):
    return XGBClassifier(
        n_estimators=n, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=SEED, n_jobs=-1
    )

def make_rf(n=400):
    return RandomForestClassifier(
        n_estimators=n, max_depth=None, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1
    )

def make_voting():
    return VotingClassifier(
        estimators=[
            ("lgbm", make_lgbm()),
            ("xgb",  make_xgb()),
            ("rf",   make_rf()),
        ],
        voting="soft", n_jobs=-1
    )

def best_threshold(probs, y_true):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.15, 0.85, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# ── load all projects ─────────────────────────────────────────────────────────

projects = {}
for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
    name = os.path.basename(f).replace(".csv", "")
    if any(ex in name.lower() for ex in EXCLUDE):
        continue
    df = load_project(f)
    if df is not None:
        projects[name] = df

print(f"Loaded {len(projects)} projects (xalan excluded)\n")

# ── main loop: transfer + fine-tune per project ───────────────────────────────

print("=" * 70)
print("  TRANSFER LEARNING + FINE-TUNE  (pre-train on others, test on target)")
print("=" * 70)

results = []

for target_name, target_df in projects.items():
    X_tgt, y_tgt = get_XY(target_df)
    X_tgt = add_features(X_tgt)

    if len(np.unique(y_tgt)) < 2 or len(y_tgt) < 20:
        print(f"  {target_name}: skipped")
        continue

    # ── split target into train/test ──────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tgt, y_tgt, test_size=0.2, random_state=SEED, stratify=y_tgt
    )
    if int(np.bincount(y_tr).min()) >= 5:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr
        )
    else:
        X_fit, X_val, y_fit, y_val = X_tr, None, y_tr, None

    # ── build transfer dataset from all OTHER projects ────────────────────
    X_other_list, y_other_list = [], []
    for other_name, other_df in projects.items():
        if other_name == target_name:
            continue
        Xo, yo = get_XY(other_df)
        Xo = add_features(Xo)
        X_other_list.append(Xo)
        y_other_list.append(yo)

    X_other = np.vstack(X_other_list)
    y_other = np.concatenate(y_other_list)

    # ── scale using combined train data ───────────────────────────────────
    scaler = StandardScaler()
    X_combined = np.vstack([X_other, X_fit])
    scaler.fit(X_combined)

    X_other_sc = np.nan_to_num(scaler.transform(X_other), nan=0, posinf=0, neginf=0)
    X_fit_sc   = np.nan_to_num(scaler.transform(X_fit),   nan=0, posinf=0, neginf=0)
    X_te_sc    = np.nan_to_num(scaler.transform(X_te),    nan=0, posinf=0, neginf=0)
    X_val_sc = (np.nan_to_num(scaler.transform(X_val), nan=0, posinf=0, neginf=0)
                if X_val is not None else None)

    # ── SMOTE on transfer data ────────────────────────────────────────────
    X_other_bal, y_other_bal = smote_safe(X_other_sc, y_other)

    # ── SMOTE on target train ─────────────────────────────────────────────
    X_tr_bal, y_tr_bal = smote_safe(X_fit_sc, y_fit)

    # ── combine: transfer data (weighted lower) + target train ────────────
    # Repeat target train 3x to give it more weight than transfer data
    X_train_final = np.vstack([X_other_bal, X_tr_bal, X_tr_bal, X_tr_bal])
    y_train_final = np.concatenate([y_other_bal, y_tr_bal, y_tr_bal, y_tr_bal])

    # ── train voting ensemble ─────────────────────────────────────────────
    model = make_voting()
    model.fit(X_train_final, y_train_final)

    # ── threshold optimization on a validation subset, never in-sample ─────
    if X_val_sc is not None:
        probs_val = model.predict_proba(X_val_sc)[:, 1]
        thr = best_threshold(probs_val, y_val)
    else:
        thr = 0.5

    # ── evaluate on test set ──────────────────────────────────────────────
    probs_te = model.predict_proba(X_te_sc)[:, 1]
    y_pred   = (probs_te >= thr).astype(int)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, probs_te) if len(np.unique(y_te)) > 1 else 0.0
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)

    results.append({
        "Project"  : target_name,
        "Samples"  : len(target_df),
        "DefRate"  : f"{y_tgt.mean():.1%}",
        "Accuracy" : round(acc, 4),
        "Precision": round(prec, 4),
        "Recall"   : round(rec, 4),
        "F1"       : round(f1, 4),
        "AUC-ROC"  : round(auc, 4),
        "Threshold": round(thr, 2),
    })

    print(f"\n  {target_name}")
    print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  thr={thr:.2f}")
    print(f"    {classification_report(y_te, y_pred, target_names=['Clean','Defective'], zero_division=0)}")

    joblib.dump({"model": model, "scaler": scaler, "threshold": thr},
                os.path.join(MODELS_DIR, f"{target_name}_model.pkl"))

# ── final report ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  FINAL RESULTS")
print("=" * 70)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))

avg_acc = res_df["Accuracy"].mean()
max_acc = res_df["Accuracy"].max()
min_acc = res_df["Accuracy"].min()

print(f"\n  Average Accuracy : {avg_acc:.4f}  ({avg_acc*100:.2f}%)")
print(f"  Best    Accuracy : {max_acc:.4f}  ({max_acc*100:.2f}%)")
print(f"  Worst   Accuracy : {min_acc:.4f}  ({min_acc*100:.2f}%)")
print(f"\n  Projects >= 95% accuracy : {(res_df['Accuracy'] >= 0.95).sum()}/{len(res_df)}")
print(f"  Projects >= 90% accuracy : {(res_df['Accuracy'] >= 0.90).sum()}/{len(res_df)}")
print(f"  Projects >= 85% accuracy : {(res_df['Accuracy'] >= 0.85).sum()}/{len(res_df)}")

res_df.to_csv(os.path.join(BASE, "results", "high_accuracy_results.csv"), index=False)
print("\n  Results saved -> results/high_accuracy_results.csv")
print("  Models  saved -> high_acc_models/")
