"""
controlled_investigation.py
============================
Controlled investigation to explain the accuracy difference between
the old pipeline (83.21%) and the corrected pipeline (59.07%).

Experiments:
  A. Correct labels + original 32 features (CK + OO + CC + engineered)
  B. Correct labels + current 19 features (CK + engineered)
  C. Correct labels + all 20 raw PROMISE metrics
  D. Correct labels + 20 raw + validated engineered features

Also fixes:
  - class_weight experiment (genuinely applies per-model weights)
  - threshold sweep (Accuracy / F1 / Balanced-Acc objectives)
  - 90% investigation
  - random-split vs LOPO comparison
"""

import os, glob, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)

from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from data_utils import normalize_bug_labels

BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "datasets")
OUTPUTS_DIR = os.path.join(BASE, "outputs");  os.makedirs(OUTPUTS_DIR, exist_ok=True)
PLOTS_DIR   = os.path.join(BASE, "plots");    os.makedirs(PLOTS_DIR, exist_ok=True)
MODELS_DIR  = os.path.join(BASE, "models");   os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# All 20 PROMISE raw metrics
CK_FEATURES  = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]
OO_FEATURES  = ["ca", "ce", "npm", "lcom3", "dam", "moa", "mfa",
                "cam", "ic", "cbm", "amc"]
CC_FEATURES  = ["max_cc", "avg_cc"]
ALL_RAW_20   = CK_FEATURES + OO_FEATURES + CC_FEATURES   # 20 features

# Engineered features used in experiment.py (original 32-feature set)
ENG_ORIG = ["log_wmc", "log_loc", "log_lcom", "log_rfc", "log_cbo",
            "wmc_per_loc", "cbo_per_loc", "rfc_per_wmc", "lcom_per_wmc",
            "coupling_sum", "complexity_avg", "log_dit"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (correct binary label: bug > 0 = 1)
# ─────────────────────────────────────────────────────────────────────────────

def load_project(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
    if not lbl:
        return None, None
    df = df.rename(columns={lbl: "bug"})
    df["bug"] = normalize_bug_labels(df["bug"])
    df = df.dropna(subset=["bug"])
    df["bug"] = df["bug"].astype(int)
    return df, os.path.basename(path).replace(".csv", "")


def load_all(exclude=("xalan",)):
    frames = []
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        if any(ex in os.path.basename(f).lower() for ex in exclude):
            continue
        df, proj = load_project(f)
        if df is None:
            continue
        df["_project"] = proj
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (row-wise only)
# ─────────────────────────────────────────────────────────────────────────────

def engineer(df):
    d = df.copy()
    eps = 1e-6
    for col in ["wmc", "loc", "lcom", "rfc", "cbo", "dit"]:
        if col in d.columns:
            d[f"log_{col}"] = np.log1p(d[col].clip(lower=0))
    if "wmc" in d.columns and "loc" in d.columns:
        d["wmc_per_loc"]  = d["wmc"]  / (d["loc"]  + eps)
    if "cbo" in d.columns and "loc" in d.columns:
        d["cbo_per_loc"]  = d["cbo"]  / (d["loc"]  + eps)
    if "rfc" in d.columns and "wmc" in d.columns:
        d["rfc_per_wmc"]  = d["rfc"]  / (d["wmc"]  + eps)
    if "lcom" in d.columns and "wmc" in d.columns:
        d["lcom_per_wmc"] = d["lcom"] / (d["wmc"]  + eps)
    if "cbo" in d.columns and "rfc" in d.columns:
        d["coupling_sum"] = d["cbo"]  + d["rfc"]
    if all(c in d.columns for c in ["wmc", "rfc", "cbo"]):
        d["complexity_avg"] = (d["wmc"] + d["rfc"] + d["cbo"]) / 3.0
    return d


def get_cols(df, fset):
    """Return available feature columns for a given feature set label."""
    if fset == "A":   # original 32: 20 raw + 12 engineered
        base = [c for c in ALL_RAW_20 if c in df.columns]
        eng  = [c for c in ENG_ORIG   if c in df.columns]
        return base + eng
    if fset == "B":   # current 19: 7 CK + 12 engineered
        base = [c for c in CK_FEATURES if c in df.columns]
        eng  = [c for c in ENG_ORIG    if c in df.columns]
        return base + eng
    if fset == "C":   # 20 raw PROMISE metrics
        return [c for c in ALL_RAW_20 if c in df.columns]
    if fset == "D":   # 20 raw + engineered
        base = [c for c in ALL_RAW_20 if c in df.columns]
        eng  = [c for c in ENG_ORIG   if c in df.columns]
        return base + eng
    return [c for c in CK_FEATURES if c in df.columns]


def prep_XY(df, fset):
    df = engineer(df)
    cols = get_cols(df, fset)
    sub = df[cols + ["bug"]].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub[c] = sub[c].fillna(sub[c].median())
    sub = sub.replace([np.inf, -np.inf], 0).drop_duplicates()
    return sub[cols].values.astype(float), sub["bug"].values.astype(int), cols


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, y_prob, name="", thr=0.5, exp="", fset="", imb=""):
    return {
        "Experiment": exp, "Features": fset, "Model": name,
        "Imbalance_Method": imb, "Threshold": round(thr, 3),
        "Accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "Precision":    round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":       round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":           round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC":      round(roc_auc_score(y_true, y_prob), 4),
        "PR_AUC":       round(average_precision_score(y_true, y_prob), 4),
        "Balanced_Acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "MCC":          round(matthews_corrcoef(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ZOO  — genuinely applies class weights where supported
# ─────────────────────────────────────────────────────────────────────────────

def model_zoo_no_balance():
    """No imbalance handling — default class weights."""
    return {
        "LogReg":     LogisticRegression(max_iter=1000, random_state=SEED),
        "DTree":      DecisionTreeClassifier(random_state=SEED),
        "RF":         RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "SVM":        SVC(probability=True, random_state=SEED),
        "XGBoost":    XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "LightGBM":   lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        "HistGB":     HistGradientBoostingClassifier(random_state=SEED),
    }


def model_zoo_class_weight(class_ratio):
    """
    Genuinely applies class_weight / scale_pos_weight to each model.
    class_ratio = n_negative / n_positive (for XGBoost scale_pos_weight).
    """
    return {
        "LogReg_CW":   LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=SEED),
        "DTree_CW":    DecisionTreeClassifier(class_weight="balanced",
                                              random_state=SEED),
        "RF_CW":       RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                              random_state=SEED, n_jobs=-1),
        "ExtraTrees_CW": ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                              random_state=SEED, n_jobs=-1),
        "SVM_CW":      SVC(probability=True, class_weight="balanced",
                           random_state=SEED),
        "XGBoost_CW":  XGBClassifier(eval_metric="logloss",
                                     scale_pos_weight=class_ratio,
                                     random_state=SEED, n_jobs=-1),
        "LightGBM_CW": lgb.LGBMClassifier(class_weight="balanced",
                                           random_state=SEED, n_jobs=-1, verbose=-1),
        "HistGB_CW":   HistGradientBoostingClassifier(
                           class_weight="balanced", random_state=SEED),
    }


def make_pipe(model, smote_type=None):
    """Build ImbPipeline. smote_type=None|'smote'|'borderline'."""
    steps = [("scaler", StandardScaler())]
    if smote_type == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    elif smote_type == "borderline":
        steps.append(("smote", BorderlineSMOTE(random_state=SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)


# ─────────────────────────────────────────────────────────────────────────────
# CV RUNNER  (SMOTE inside folds, class-weight applied per model)
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(models_dict, X, y, skf, smote_type=None, imb_label="none"):
    rows = []
    for name, model in models_dict.items():
        pipe = make_pipe(model, smote_type=smote_type)
        accs, f1s, aucs, praucs, bals, recs, precs = [], [], [], [], [], [], []
        for tr_i, va_i in skf.split(X, y):
            pipe.fit(X[tr_i], y[tr_i])
            yp  = pipe.predict(X[va_i])
            ypr = pipe.predict_proba(X[va_i])[:, 1]
            accs.append(accuracy_score(y[va_i], yp))
            f1s.append(f1_score(y[va_i], yp, zero_division=0))
            aucs.append(roc_auc_score(y[va_i], ypr))
            praucs.append(average_precision_score(y[va_i], ypr))
            bals.append(balanced_accuracy_score(y[va_i], yp))
            recs.append(recall_score(y[va_i], yp, zero_division=0))
            precs.append(precision_score(y[va_i], yp, zero_division=0))
        rows.append({
            "Model": name, "Imbalance": imb_label,
            "CV_Acc":  round(np.mean(accs), 4),
            "CV_F1":   round(np.mean(f1s),  4),
            "CV_F1_Std": round(np.std(f1s), 4),
            "CV_AUC":  round(np.mean(aucs), 4),
            "CV_PRAUC":round(np.mean(praucs),4),
            "CV_BalAcc":round(np.mean(bals), 4),
            "CV_Rec":  round(np.mean(recs),  4),
            "CV_Prec": round(np.mean(precs), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SWEEP  (OOF predictions — test set never touched)
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(pipe, X_train, y_train, skf):
    oof = cross_val_predict(pipe, X_train, y_train,
                            cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
    rows = []
    for t in np.arange(0.10, 0.91, 0.05):
        yp = (oof >= t).astype(int)
        rows.append({
            "Threshold":    round(t, 2),
            "Accuracy":     round(accuracy_score(y_train, yp), 4),
            "Precision":    round(precision_score(y_train, yp, zero_division=0), 4),
            "Recall":       round(recall_score(y_train, yp, zero_division=0), 4),
            "F1":           round(f1_score(y_train, yp, zero_division=0), 4),
            "Balanced_Acc": round(balanced_accuracy_score(y_train, yp), 4),
            "MCC":          round(matthews_corrcoef(y_train, yp), 4),
        })
    df = pd.DataFrame(rows)
    best_acc  = float(df.loc[df["Accuracy"].idxmax(),     "Threshold"])
    best_f1   = float(df.loc[df["F1"].idxmax(),           "Threshold"])
    best_bal  = float(df.loc[df["Balanced_Acc"].idxmax(), "Threshold"])
    return df, best_acc, best_f1, best_bal, oof


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_models(X_train, y_train, skf):
    lgbm_grid = {
        "model__n_estimators":      [200, 400, 600],
        "model__max_depth":         [3, 5, 7, -1],
        "model__learning_rate":     [0.01, 0.05, 0.1],
        "model__num_leaves":        [31, 63, 127],
        "model__min_child_samples": [5, 10, 20],
        "model__subsample":         [0.7, 0.8, 1.0],
        "model__colsample_bytree":  [0.7, 0.8, 1.0],
        "model__reg_alpha":         [0, 0.1, 0.5],
        "model__reg_lambda":        [0.1, 1.0, 5.0],
    }
    xgb_grid = {
        "model__n_estimators":     [200, 400, 600],
        "model__max_depth":        [3, 5, 7],
        "model__learning_rate":    [0.01, 0.05, 0.1],
        "model__subsample":        [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_alpha":        [0, 0.1, 1.0],
        "model__reg_lambda":       [1.0, 2.0, 5.0],
    }
    et_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth":    [None, 10, 20, 30],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__min_samples_leaf": [1, 2, 4],
    }
    brf_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth":    [None, 10, 20],
        "model__max_features": ["sqrt", "log2", 0.5],
    }

    tuned = {}
    configs = [
        ("LightGBM_Tuned",
         lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
         lgbm_grid, "smote"),
        ("XGBoost_Tuned",
         XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
         xgb_grid, "smote"),
        ("ExtraTrees_Tuned",
         ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
         et_grid, "smote"),
    ]
    for mname, base_model, grid, stype in configs:
        pipe = make_pipe(base_model, smote_type=stype)
        rs = RandomizedSearchCV(pipe, grid, n_iter=40, cv=skf, scoring="f1",
                                random_state=SEED, n_jobs=-1, refit=True)
        rs.fit(X_train, y_train)
        tuned[mname] = rs.best_estimator_
        print(f"  {mname}: CV F1={rs.best_score_:.4f}")

    # BalancedRF (no SMOTE — already balanced internally)
    brf_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("model", BalancedRandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ])
    rs_brf = RandomizedSearchCV(brf_pipe, brf_grid, n_iter=20, cv=skf, scoring="f1",
                                random_state=SEED, n_jobs=-1, refit=True)
    rs_brf.fit(X_train, y_train)
    tuned["BalancedRF_Tuned"] = rs_brf.best_estimator_
    print(f"  BalancedRF_Tuned: CV F1={rs_brf.best_score_:.4f}")

    # Soft-voting ensemble of top 3
    lgbm_best = tuned["LightGBM_Tuned"].named_steps["model"]
    xgb_best  = tuned["XGBoost_Tuned"].named_steps["model"]
    et_best   = tuned["ExtraTrees_Tuned"].named_steps["model"]
    ensemble_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=SEED)),
        ("model",  VotingClassifier(
            estimators=[("lgbm", lgbm_best), ("xgb", xgb_best), ("et", et_best)],
            voting="soft", n_jobs=-1
        )),
    ])
    ensemble_pipe.fit(X_train, y_train)
    tuned["Ensemble_Voting"] = ensemble_pipe
    print(f"  Ensemble_Voting: fitted")

    return tuned


# ─────────────────────────────────────────────────────────────────────────────
# LEAVE-ONE-PROJECT-OUT
# ─────────────────────────────────────────────────────────────────────────────

def lopo(fset="D", exclude=("xalan",)):
    all_projects = {}
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        pname = os.path.basename(f).replace(".csv", "")
        if any(ex in pname.lower() for ex in exclude):
            continue
        df, _ = load_project(f)
        if df is None:
            continue
        df = engineer(df)
        all_projects[pname] = df

    rows = []
    for test_proj, test_df in all_projects.items():
        train_df = pd.concat(
            [df for p, df in all_projects.items() if p != test_proj],
            ignore_index=True
        )
        cols = get_cols(train_df, fset)
        cols = [c for c in cols if c in test_df.columns]

        def prep(df):
            sub = df[cols + ["bug"]].copy()
            for c in cols:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
                sub[c] = sub[c].fillna(sub[c].median())
            return sub.replace([np.inf, -np.inf], 0).drop_duplicates()

        tr = prep(train_df); te = prep(test_df)
        X_tr = tr[cols].values.astype(float); y_tr = tr["bug"].values.astype(int)
        X_te = te[cols].values.astype(float); y_te = te["bug"].values.astype(int)

        if len(np.unique(y_te)) < 2:
            continue

        minority = int(np.bincount(y_tr).min())
        k = min(5, minority - 1)
        smote_type = "smote" if k >= 1 else None
        pipe = make_pipe(
            lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               random_state=SEED, n_jobs=-1, verbose=-1),
            smote_type=smote_type
        )
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        row = metrics(y_te, y_pred, y_prob, name=test_proj, thr=0.5,
                      exp="LOPO", fset=fset, imb="SMOTE")
        row["Test_Project"] = test_proj
        row["Test_Samples"] = len(y_te)
        row["Defect_Pct"]   = round(100 * y_te.mean(), 1)
        rows.append(row)
        print(f"  LOPO {test_proj:20s}  Acc={row['Accuracy']:.4f}"
              f"  F1={row['F1']:.4f}  AUC={row['ROC_AUC']:.4f}")

    return pd.DataFrame(rows)
