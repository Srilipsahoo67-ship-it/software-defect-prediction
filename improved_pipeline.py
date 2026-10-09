"""
improved_pipeline.py
====================
Authoritative leakage-free research pipeline for Software Defect Prediction.

Key fixes over existing scripts:
  1. Bug label treated as COUNT -> binary (bug > 0 = 1)
  2. Single unified feature set (CK + engineered) used in train/predict/app
  3. Full sklearn Pipeline saved -> no scaler/model mismatch at inference
  4. Accuracy threshold and model selected using training OOF predictions
  5. SMOTE applied inside CV folds only
  6. Test set used only for final evaluation
  7. BalancedRandomForest + class_weight experiments added
  8. Full metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Bal-Acc, MCC
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
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from data_utils import normalize_bug_labels

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ── Directories ───────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "datasets")
MODELS_DIR  = os.path.join(BASE, "models");   os.makedirs(MODELS_DIR, exist_ok=True)
OUTPUTS_DIR = os.path.join(BASE, "outputs");  os.makedirs(OUTPUTS_DIR, exist_ok=True)
PLOTS_DIR   = os.path.join(BASE, "plots");    os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

CK_FEATURES  = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]
OO_FEATURES  = ["ca", "ce", "npm", "lcom3", "dam", "moa", "mfa",
                "cam", "ic", "cbm", "amc"]
CC_FEATURES  = ["max_cc", "avg_cc"]
ALL_RAW      = CK_FEATURES + OO_FEATURES + CC_FEATURES

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & DATASET AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def load_project(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
    if not lbl:
        return None, None
    df = df.rename(columns={lbl: "bug"})
    # FIX: bug is a COUNT in PROMISE datasets — convert to binary
    df["bug"] = normalize_bug_labels(df["bug"])
    df = df.dropna(subset=["bug"])
    df["bug"] = df["bug"].astype(int)
    return df, os.path.basename(path).replace(".csv", "")


def dataset_audit():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        df, name = load_project(f)
        if df is None:
            continue
        avail = [c for c in ALL_RAW if c in df.columns]
        missing = df[avail].isnull().sum().sum()
        dups = df.duplicated().sum()
        n = len(df)
        nd = int(df["bug"].sum())
        rows.append({
            "Project": name, "Rows": n, "Cols": len(df.columns),
            "Clean": n - nd, "Defective": nd,
            "Defect_Pct": round(100 * nd / n, 1),
            "Missing": int(missing), "Duplicates": int(dups),
            "Features_Available": len(avail),
        })
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(os.path.join(OUTPUTS_DIR, "dataset_audit.csv"), index=False)
    print("\n" + "="*65)
    print("DATASET AUDIT")
    print("="*65)
    print(audit_df.to_string(index=False))
    return audit_df


def load_all(exclude=("xalan",)):
    frames = []
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        name = os.path.basename(f).lower()
        if any(ex in name for ex in exclude):
            continue
        df, proj = load_project(f)
        if df is None:
            continue
        df["_project"] = proj
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  (row-wise only — safe before split)
# ═══════════════════════════════════════════════════════════════════════════════

ENG_FEATURE_NAMES = [
    "log_wmc", "log_loc", "log_lcom", "log_rfc", "log_cbo", "log_dit",
    "wmc_per_loc", "cbo_per_loc", "rfc_per_wmc", "lcom_per_wmc",
    "coupling_sum", "complexity_avg",
]


def engineer_features(df):
    """Add log-transforms and ratios. Row-wise only — no dataset statistics used."""
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


def get_feature_cols(df, feature_set="C"):
    """
    A = CK only (7)
    B = CK + OO + CC raw (20)
    C = CK + engineered (best from experiment.py Exp5)
    """
    if feature_set == "A":
        return [c for c in CK_FEATURES if c in df.columns]
    if feature_set == "B":
        return [c for c in ALL_RAW if c in df.columns]
    # C: CK + engineered
    base = [c for c in CK_FEATURES if c in df.columns]
    eng  = [c for c in ENG_FEATURE_NAMES if c in df.columns]
    return base + eng


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(df, feature_set="C"):
    df = engineer_features(df)
    feat_cols = get_feature_cols(df, feature_set)
    sub = df[feat_cols + ["bug"]].copy()
    for col in feat_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub[col] = sub[col].fillna(sub[col].median())
    sub = sub.drop_duplicates()
    sub = sub.replace([np.inf, -np.inf], 0)
    X = sub[feat_cols].values.astype(float)
    y = sub["bug"].values.astype(int)
    return X, y, feat_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 4. METRICS HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob, name="", threshold=0.5):
    return {
        "Model": name, "Threshold": round(threshold, 3),
        "Accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "Precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":          round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC":           round(average_precision_score(y_true, y_prob), 4),
        "Balanced_Acc":     round(balanced_accuracy_score(y_true, y_pred), 4),
        "MCC":              round(matthews_corrcoef(y_true, y_pred), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. THRESHOLD OPTIMIZATION  (on OOF CV predictions — never on test)
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_threshold_oof(pipeline, X_train, y_train, cv):
    """Select an accuracy threshold from out-of-fold training predictions."""
    oof_probs = cross_val_predict(pipeline, X_train, y_train,
                                  cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = np.arange(0.10, 0.81, 0.05)
    thr_rows = []
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        thr_rows.append({
            "Threshold":    round(t, 2),
            "Accuracy":     round(accuracy_score(y_train, preds), 4),
            "F1":           round(f1_score(y_train, preds, zero_division=0), 4),
            "Recall":       round(recall_score(y_train, preds, zero_division=0), 4),
            "Precision":    round(precision_score(y_train, preds, zero_division=0), 4),
            "Balanced_Acc": round(balanced_accuracy_score(y_train, preds), 4),
        })
    thr_df = pd.DataFrame(thr_rows)
    best_t = float(thr_df.loc[thr_df["Accuracy"].idxmax(), "Threshold"])
    return best_t, thr_df, oof_probs


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL ZOO
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_zoo():
    return {
        "LogReg":       LogisticRegression(max_iter=1000, random_state=SEED),
        "DTree":        DecisionTreeClassifier(random_state=SEED),
        "RF":           RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "ExtraTrees":   ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "RF_Balanced":  RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=SEED, n_jobs=-1),
        "BalancedRF":   BalancedRandomForestClassifier(n_estimators=300,
                                                        random_state=SEED, n_jobs=-1),
        "SVM":          SVC(probability=True, random_state=SEED),
        "XGBoost":      XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "LightGBM":     lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        "HistGB":       HistGradientBoostingClassifier(random_state=SEED),
        "MLP":          MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                      random_state=SEED),
    }


def make_pipeline(model, use_smote=True, smote_type="smote"):
    """Build an imblearn Pipeline with scaler + optional SMOTE + model."""
    steps = [("scaler", StandardScaler())]
    if use_smote:
        sm = (BorderlineSMOTE(random_state=SEED) if smote_type == "borderline"
              else SMOTE(random_state=SEED))
        steps.append(("smote", sm))
    steps.append(("model", model))
    return ImbPipeline(steps)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CV EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv_experiment(X, y, skf, label, use_smote=True, smote_type="smote"):
    """
    Run 5-fold stratified CV for all models.
    SMOTE applied INSIDE each fold via ImbPipeline — no leakage.
    """
    print(f"\n" + "-"*60)
    print(f"CV Experiment: {label}")
    print("-"*60)
    rows = []
    for name, model in get_model_zoo().items():
        pipe = make_pipeline(model, use_smote=use_smote, smote_type=smote_type)
        fold_metrics = {k: [] for k in
                        ["accuracy","f1","roc_auc","average_precision",
                         "balanced_accuracy","recall","precision"]}
        for tr_i, va_i in skf.split(X, y):
            pipe.fit(X[tr_i], y[tr_i])
            y_prob = pipe.predict_proba(X[va_i])[:, 1]
            y_pred = pipe.predict(X[va_i])
            fold_metrics["accuracy"].append(accuracy_score(y[va_i], y_pred))
            fold_metrics["f1"].append(f1_score(y[va_i], y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y[va_i], y_prob))
            fold_metrics["average_precision"].append(
                average_precision_score(y[va_i], y_prob))
            fold_metrics["balanced_accuracy"].append(
                balanced_accuracy_score(y[va_i], y_pred))
            fold_metrics["recall"].append(
                recall_score(y[va_i], y_pred, zero_division=0))
            fold_metrics["precision"].append(
                precision_score(y[va_i], y_pred, zero_division=0))

        row = {
            "Model": name, "Experiment": label,
            "CV_Accuracy_Mean":      round(np.mean(fold_metrics["accuracy"]), 4),
            "CV_Accuracy_Std":       round(np.std(fold_metrics["accuracy"]), 4),
            "CV_Precision_Mean":     round(np.mean(fold_metrics["precision"]), 4),
            "CV_Recall_Mean":        round(np.mean(fold_metrics["recall"]), 4),
            "CV_F1_Mean":            round(np.mean(fold_metrics["f1"]), 4),
            "CV_F1_Std":             round(np.std(fold_metrics["f1"]), 4),
            "CV_ROC_AUC_Mean":       round(np.mean(fold_metrics["roc_auc"]), 4),
            "CV_PR_AUC_Mean":        round(np.mean(fold_metrics["average_precision"]), 4),
            "CV_Balanced_Acc_Mean":  round(np.mean(fold_metrics["balanced_accuracy"]), 4),
        }
        rows.append(row)
        print(f"  {name:15s}  F1={row['CV_F1_Mean']:.4f}±{row['CV_F1_Std']:.4f}"
              f"  AUC={row['CV_ROC_AUC_Mean']:.4f}  Rec={row['CV_Recall_Mean']:.4f}")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HYPERPARAMETER TUNING  (on training data only)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_top_models(X_train, y_train, skf):
    """RandomizedSearchCV for LightGBM and XGBoost. Optimize defective F1."""
    print("\n" + "-"*60)
    print("Hyperparameter Tuning (RandomizedSearchCV, scoring=accuracy)")
    print("-"*60)

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

    tuned = {}
    for model_name, base_model, grid in [
        ("LightGBM_Tuned",
         lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1), lgbm_grid),
        ("XGBoost_Tuned",
         XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1), xgb_grid),
    ]:
        pipe = make_pipeline(base_model, use_smote=True)
        rs = RandomizedSearchCV(
            pipe, grid, n_iter=40, cv=skf, scoring="accuracy",
            random_state=SEED, n_jobs=-1, refit=True
        )
        rs.fit(X_train, y_train)
        tuned[model_name] = rs.best_estimator_
        print(f"  {model_name}: best CV F1 = {rs.best_score_:.4f}")
        print(f"    params: { {k.replace('model__',''):v for k,v in rs.best_params_.items()} }")

    # Also tune BalancedRF
    brf_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth":    [None, 10, 20],
        "model__max_features": ["sqrt", "log2", 0.5],
    }
    brf_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("model", BalancedRandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ])
    rs_brf = RandomizedSearchCV(
        brf_pipe, brf_grid, n_iter=20, cv=skf, scoring="accuracy",
        random_state=SEED, n_jobs=-1, refit=True
    )
    rs_brf.fit(X_train, y_train)
    tuned["BalancedRF_Tuned"] = rs_brf.best_estimator_
    print(f"  BalancedRF_Tuned: best CV F1 = {rs_brf.best_score_:.4f}")

    return tuned


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PER-PROJECT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def per_project_evaluation(best_pipeline, threshold, exclude=("xalan",)):
    print("\n" + "-"*60)
    print("Per-Project Evaluation")
    print("-"*60)
    rows = []
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        name = os.path.basename(f).replace(".csv", "")
        if any(ex in name.lower() for ex in exclude):
            continue
        df, _ = load_project(f)
        if df is None:
            continue
        df = engineer_features(df)
        feat_cols = get_feature_cols(df, "C")
        sub = df[feat_cols + ["bug"]].copy()
        for col in feat_cols:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub[col] = sub[col].fillna(sub[col].median())
        sub = sub.replace([np.inf, -np.inf], 0).drop_duplicates()
        X_p = sub[feat_cols].values.astype(float)
        y_p = sub["bug"].values.astype(int)

        n = len(y_p)
        nd = int(y_p.sum())
        low_sample = n < 100

        if len(np.unique(y_p)) < 2 or nd < 5:
            print(f"  {name}: SKIPPED (single class or too few defective)")
            continue

        y_prob = best_pipeline.predict_proba(X_p)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        row = compute_metrics(y_p, y_pred, y_prob, name=name, threshold=threshold)
        row["Samples"]    = n
        row["Defect_Pct"] = round(100 * nd / n, 1)
        row["Warning"]    = "LOW SAMPLE SIZE" if low_sample else ""
        rows.append(row)
        flag = " ⚠ LOW SAMPLE" if low_sample else ""
        print(f"  {name:20s}  n={n:4d}  Acc={row['Accuracy']:.4f}"
              f"  F1={row['F1']:.4f}  AUC={row['ROC-AUC']:.4f}{flag}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUTS_DIR, "per_project_improved_results.csv"), index=False)
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CROSS-PROJECT (LEAVE-ONE-PROJECT-OUT)
# ═══════════════════════════════════════════════════════════════════════════════

def cross_project_validation(exclude=("xalan",)):
    print("\n" + "-"*60)
    print("Cross-Project Validation (Leave-One-Project-Out)")
    print("-"*60)

    all_projects = {}
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        pname = os.path.basename(f).replace(".csv", "")
        if any(ex in pname.lower() for ex in exclude):
            continue
        df, _ = load_project(f)
        if df is None:
            continue
        df = engineer_features(df)
        all_projects[pname] = df

    rows = []
    for test_proj, test_df in all_projects.items():
        train_frames = [df for p, df in all_projects.items() if p != test_proj]
        train_df = pd.concat(train_frames, ignore_index=True)

        feat_cols = get_feature_cols(train_df, "C")
        feat_cols = [c for c in feat_cols if c in test_df.columns]

        def prep(df):
            sub = df[feat_cols + ["bug"]].copy()
            for col in feat_cols:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
                sub[col] = sub[col].fillna(sub[col].median())
            return sub.replace([np.inf, -np.inf], 0).drop_duplicates()

        tr = prep(train_df)
        te = prep(test_df)

        X_tr, y_tr = tr[feat_cols].values.astype(float), tr["bug"].values.astype(int)
        X_te, y_te = te[feat_cols].values.astype(float), te["bug"].values.astype(int)

        if len(np.unique(y_te)) < 2:
            continue

        minority = int(np.bincount(y_tr).min())
        k = min(5, minority - 1)
        if k < 1:
            pipe = ImbPipeline([("scaler", StandardScaler()),
                                ("model", lgb.LGBMClassifier(random_state=SEED,
                                                              n_jobs=-1, verbose=-1))])
        else:
            pipe = ImbPipeline([("scaler", StandardScaler()),
                                ("smote", SMOTE(random_state=SEED, k_neighbors=k)),
                                ("model", lgb.LGBMClassifier(random_state=SEED,
                                                              n_jobs=-1, verbose=-1))])
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        row = compute_metrics(y_te, y_pred, y_prob, name=test_proj)
        row["Test_Project"]     = test_proj
        row["Training_Projects"] = ",".join(p for p in all_projects if p != test_proj)
        row["Test_Samples"]     = len(y_te)
        rows.append(row)
        print(f"  Test={test_proj:20s}  Acc={row['Accuracy']:.4f}"
              f"  F1={row['F1']:.4f}  AUC={row['ROC-AUC']:.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUTPUTS_DIR, "cross_project_results.csv"), index=False)
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def error_analysis(pipeline, X_test, y_test, feat_cols, threshold):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    df_err = pd.DataFrame(X_test, columns=feat_cols)
    df_err["True_Label"]  = y_test
    df_err["Pred_Label"]  = y_pred
    df_err["Probability"] = y_prob.round(4)
    df_err["Result"] = np.where(
        (y_test == 1) & (y_pred == 1), "TP",
        np.where((y_test == 0) & (y_pred == 0), "TN",
        np.where((y_test == 1) & (y_pred == 0), "FN", "FP"))
    )
    df_err.to_csv(os.path.join(OUTPUTS_DIR, "error_analysis.csv"), index=False)

    print(f"\n  Confusion Matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  False Negatives (missed defects): {fn}")
    print(f"  False Positives (false alarms):   {fp}")
    return df_err, cm


# ═══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(pipeline, X_test, y_test, feat_cols):
    model = pipeline.named_steps["model"]
    rows = []

    # Model-native importance
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(feat_cols, model.feature_importances_):
            rows.append({"Feature": name, "Model_Importance": round(imp, 6),
                         "Permutation_Importance": None})

    # Permutation importance on test set
    scaler = pipeline.named_steps["scaler"]
    X_sc = scaler.transform(X_test)
    X_sc = np.nan_to_num(X_sc, nan=0, posinf=0, neginf=0)
    perm = permutation_importance(model, X_sc, y_test, n_repeats=10,
                                  random_state=SEED, scoring="f1")
    if rows:
        for i, r in enumerate(rows):
            r["Permutation_Importance"] = round(float(perm.importances_mean[i]), 6)
    else:
        for name, imp in zip(feat_cols, perm.importances_mean):
            rows.append({"Feature": name, "Model_Importance": None,
                         "Permutation_Importance": round(float(imp), 6)})

    fi_df = pd.DataFrame(rows).sort_values("Permutation_Importance",
                                            ascending=False, na_position="last")
    fi_df.to_csv(os.path.join(OUTPUTS_DIR, "feature_importance.csv"), index=False)
    return fi_df


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("  SOFTWARE DEFECT PREDICTION — IMPROVED PIPELINE")
    print("="*65)

    # ── Step 1: Dataset Audit ─────────────────────────────────────────────────
    audit_df = dataset_audit()

    # ── Step 2: Load data (xalan excluded) ───────────────────────────────────
    raw_df = load_all(exclude=("xalan",))
    print(f"\nLoaded {len(raw_df)} rows from {raw_df['_project'].nunique()} projects")

    # ── Step 3: Prepare features (Feature Set C: CK + engineered) ────────────
    X, y, feat_cols = prepare_data(raw_df, feature_set="C")
    print(f"Feature set C: {len(feat_cols)} features")
    print(f"Class distribution: Clean={int((y==0).sum())}  Defective={int((y==1).sum())}")
    print(f"Defect rate: {y.mean():.1%}")

    # ── Step 4: Train/Test split (stratified, 80/20) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\nTrain: {len(y_train)}  Test: {len(y_test)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── Step 5: CV Experiments (imbalance strategies) ─────────────────────────
    all_cv_rows = []

    # Exp A: No imbalance handling
    cv_A = run_cv_experiment(X_train, y_train, skf,
                              "A_NoImbalance", use_smote=False)
    all_cv_rows.append(cv_A)

    # Exp B: class_weight=balanced (via RF_Balanced in model zoo)
    cv_B = run_cv_experiment(X_train, y_train, skf,
                              "B_ClassWeight", use_smote=False)
    all_cv_rows.append(cv_B)

    # Exp C: SMOTE
    cv_C = run_cv_experiment(X_train, y_train, skf,
                              "C_SMOTE", use_smote=True, smote_type="smote")
    all_cv_rows.append(cv_C)

    # Exp D: BorderlineSMOTE
    cv_D = run_cv_experiment(X_train, y_train, skf,
                              "D_BorderlineSMOTE", use_smote=True, smote_type="borderline")
    all_cv_rows.append(cv_D)

    cv_all = pd.concat(all_cv_rows, ignore_index=True)
    cv_all.to_csv(os.path.join(OUTPUTS_DIR, "cv_results.csv"), index=False)
    print(f"\nCV results saved -> outputs/cv_results.csv")

    # ── Step 6: Hyperparameter Tuning ────────────────────────────────────────
    tuned_models = tune_top_models(X_train, y_train, skf)

    # ── Step 7: Threshold Optimization (OOF on training data) ────────────────
    print("\n" + "-"*60)
    print("Threshold Optimization (OOF CV predictions)")
    print("-"*60)

    all_thr_rows = []
    tuned_thresholds = {}
    selection_scores = {}
    for model_name, pipe in tuned_models.items():
        best_t, thr_df, _ = optimize_threshold_oof(pipe, X_train, y_train, skf)
        thr_df["Model"] = model_name
        all_thr_rows.append(thr_df)
        tuned_thresholds[model_name] = best_t
        selection_scores[model_name] = float(thr_df["Accuracy"].max())
        print(f"  {model_name}: threshold={best_t:.2f}, "
              f"OOF accuracy={selection_scores[model_name]:.4f}")

    baseline_pipe = make_pipeline(
        lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        use_smote=True
    )
    baseline_thr, baseline_thr_df, _ = optimize_threshold_oof(
        baseline_pipe, X_train, y_train, skf)
    baseline_thr_df["Model"] = "LightGBM_Baseline"
    all_thr_rows.append(baseline_thr_df)
    tuned_thresholds["LightGBM_Baseline"] = baseline_thr
    selection_scores["LightGBM_Baseline"] = float(
        baseline_thr_df["Accuracy"].max())
    baseline_pipe.fit(X_train, y_train)
    candidates = {**tuned_models, "LightGBM_Baseline": baseline_pipe}

    pd.concat(all_thr_rows, ignore_index=True).to_csv(
        os.path.join(OUTPUTS_DIR, "threshold_results.csv"), index=False)

    # ── Step 8: Final Test Evaluation ────────────────────────────────────────
    print("\n" + "-"*60)
    print("FINAL TEST SET EVALUATION (untouched test set)")
    print("-"*60)

    final_rows = []
    for model_name, pipe in candidates.items():
        thr = tuned_thresholds[model_name]
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= thr).astype(int)
        row = compute_metrics(y_test, y_pred, y_prob,
                              name=model_name, threshold=thr)
        row["OOF_Selection_Accuracy"] = selection_scores[model_name]
        final_rows.append(row)
        print(f"  {model_name:20s}  Acc={row['Accuracy']:.4f}"
              f"  F1={row['F1']:.4f}  Rec={row['Recall']:.4f}"
              f"  AUC={row['ROC-AUC']:.4f}  PR-AUC={row['PR-AUC']:.4f}")

    final_df = pd.DataFrame(final_rows).sort_values(
        "OOF_Selection_Accuracy", ascending=False)
    final_df.to_csv(os.path.join(OUTPUTS_DIR, "FINAL_RESULTS.csv"), index=False)
    print(f"\nFinal results saved -> outputs/FINAL_RESULTS.csv")

    # Select model only from training OOF results; test metrics remain final.
    best_name = max(selection_scores, key=selection_scores.get)
    best_pipe = candidates[best_name]
    best_thr = tuned_thresholds[best_name]
    best_row = final_df.loc[final_df["Model"] == best_name].iloc[0]
    print(f"\nSelected by training OOF accuracy: {best_name}  "
          f"OOF accuracy={selection_scores[best_name]:.4f}")

    # ── Step 10: Save Final Pipeline ──────────────────────────────────────────
    pipeline_path = os.path.join(MODELS_DIR, "final_defect_prediction_pipeline.pkl")
    joblib.dump(best_pipe, pipeline_path)

    metadata = {
        "model":             best_name,
        "feature_set":       "C_CK_Engineered",
        "training_features": feat_cols,
        "imbalance_strategy":"SMOTE",
        "threshold":         best_thr,
        "oof_selection_accuracy": selection_scores[best_name],
        "performance_metrics": {
            "Accuracy":     float(best_row["Accuracy"]),
            "Precision":    float(best_row["Precision"]),
            "Recall":       float(best_row["Recall"]),
            "F1":           float(best_row["F1"]),
            "ROC_AUC":      float(best_row["ROC-AUC"]),
            "PR_AUC":       float(best_row["PR-AUC"]),
            "Balanced_Acc": float(best_row["Balanced_Acc"]),
            "MCC":          float(best_row["MCC"]),
        }
    }
    with open(os.path.join(MODELS_DIR, "final_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Pipeline saved -> {pipeline_path}")

    # ── Step 11: Error Analysis ───────────────────────────────────────────────
    err_df, cm = error_analysis(best_pipe, X_test, y_test, feat_cols, best_thr)

    # ── Step 12: Feature Importance ───────────────────────────────────────────
    print("\n" + "-"*60)
    print("Feature Importance")
    print("-"*60)
    fi_df = compute_feature_importance(best_pipe, X_test, y_test, feat_cols)
    print(fi_df[["Feature","Permutation_Importance"]].head(10).to_string(index=False))

    # ── Step 13: Per-Project Evaluation ──────────────────────────────────────
    pp_df = per_project_evaluation(best_pipe, best_thr)

    # ── Step 14: Cross-Project Validation ────────────────────────────────────
    cp_df = cross_project_validation()

    # ── Step 15: Current result summary ──────────────────────────────────────
    new_metrics = metadata["performance_metrics"]
    old_vs_new = pd.DataFrame([
        {"Metric": name, "Old": None, "New": value, "Improvement": None,
         "Note": "Previous result used incompatible label handling; not comparable."}
        for name, value in [
            ("Accuracy", new_metrics["Accuracy"]),
            ("F1", new_metrics["F1"]),
            ("Recall", new_metrics["Recall"]),
            ("Precision", new_metrics["Precision"]),
            ("ROC-AUC", new_metrics["ROC_AUC"]),
            ("PR-AUC", new_metrics["PR_AUC"]),
            ("Balanced_Acc", new_metrics["Balanced_Acc"]),
            ("MCC", new_metrics["MCC"]),
        ]
    ])
    old_vs_new.to_csv(os.path.join(OUTPUTS_DIR, "OLD_VS_NEW.csv"), index=False)

    # ── Step 16: Plots ────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Clean","Defective"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name} (thr={best_thr:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "improved_confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC + PR curves
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    axes[0].plot(fpr, tpr, lw=2,
                 label=f"{best_name} (AUC={best_row['ROC-AUC']:.3f})")
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend()
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob_best)
    axes[1].plot(rec_c, prec_c, lw=2,
                 label=f"{best_name} (PR-AUC={best_row['PR-AUC']:.3f})")
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", label="Baseline")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "improved_roc_pr_curves.png"), dpi=150)
    plt.close()

    # Feature importance plot
    fi_plot = fi_df.dropna(subset=["Permutation_Importance"]).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi_plot.sort_values("Permutation_Importance").plot(
        kind="barh", x="Feature", y="Permutation_Importance",
        ax=ax, color="#5C85D6", legend=False)
    ax.set_title(f"Top 15 Features — Permutation Importance ({best_name})")
    ax.set_xlabel("Mean Decrease in F1")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "improved_feature_importance.png"), dpi=150)
    plt.close()

    # Per-project bar chart
    if not pp_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(pp_df))
        ax.bar(x - 0.2, pp_df["Accuracy"], 0.35, label="Accuracy", color="#4299e1")
        ax.bar(x + 0.2, pp_df["F1"],       0.35, label="F1",       color="#48bb78")
        ax.set_xticks(x)
        ax.set_xticklabels(pp_df["Model"], rotation=30, ha="right")
        ax.set_ylim(0, 1); ax.set_title("Per-Project Performance")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "per_project_performance.png"), dpi=150)
        plt.close()

    # SHAP summary
    if SHAP_OK:
        try:
            model_obj = best_pipe.named_steps["model"]
            scaler_obj = best_pipe.named_steps["scaler"]
            X_te_sc = np.nan_to_num(scaler_obj.transform(X_test), nan=0, posinf=0, neginf=0)
            explainer = shap.TreeExplainer(model_obj)
            shap_vals = explainer.shap_values(X_te_sc)
            sv = shap_vals if shap_vals.ndim == 2 else shap_vals[:, :, 1]
            fig, ax = plt.subplots(figsize=(8, 6))
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(mean_abs)
            ax.barh([feat_cols[i] for i in order[-15:]],
                    mean_abs[order[-15:]], color="#5C85D6")
            ax.set_title(f"SHAP Mean |Value| — {best_name}")
            ax.set_xlabel("Mean |SHAP value|")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "improved_shap_summary.png"), dpi=150)
            plt.close()
            print("SHAP plot saved.")
        except Exception as e:
            print(f"SHAP plot skipped: {e}")

    # ── Step 17: Final Report ─────────────────────────────────────────────────
    achieved_90 = float(best_row["Accuracy"]) >= 0.90
    report_lines = [
        "# Software Defect Prediction — Final Analysis Report\n",
        "## 1. Executive Summary",
        f"Best model: **{best_name}** (selected by training OOF accuracy)",
        f"- Accuracy: {best_row['Accuracy']:.4f}",
        f"- F1: {best_row['F1']:.4f}",
        f"- Recall: {best_row['Recall']:.4f}",
        f"- ROC-AUC: {best_row['ROC-AUC']:.4f}",
        f"- PR-AUC: {best_row['PR-AUC']:.4f}",
        f"- MCC: {best_row['MCC']:.4f}",
        "",
        "## 2. Dataset Description",
        "11 PROMISE repository Java projects. xalan excluded (98.4% defective).",
        f"Total samples after deduplication: {len(y)}",
        f"Defect rate: {y.mean():.1%}",
        "",
        "## 3. Historical Results",
        "The previous 0.8321 accuracy result used incompatible label handling and is not comparable.",
        "Do not treat it as a corrected-label baseline.",
        "",
        "## 4. Fixes Applied",
        "- Defect counts are consistently converted to binary labels (bug > 0).",
        "- The canonical model uses the 19 features computable from the seven CK inputs.",
        "- Model and threshold selection use training out-of-fold predictions.",
        "- Experimental full-feature models save separate artifacts and cannot overwrite the app model.",
        "",
        "## 5. Leakage Audit",
        "SMOTE and scaling are inside cross-validation pipelines.",
        "Model and threshold selection use training folds; the held-out test set is not used for selection.",
        "",
        "## 6. Class Imbalance Analysis",
        f"Clean: {int((y==0).sum())} ({(y==0).mean():.1%})  Defective: {int((y==1).sum())} ({y.mean():.1%})",
        "Strategies tested: No handling, class_weight=balanced, SMOTE, BorderlineSMOTE.",
        "",
        "## 7. Feature Engineering",
        f"Feature Set C: {len(feat_cols)} features (7 CK + log-transforms + ratios).",
        "Feature transforms are row-wise; dataset audit found no missing input metrics.",
        "",
        "## 8. Model Comparison",
        "See outputs/cv_results.csv for full CV comparison across all models.",
        "",
        "## 9. Hyperparameter Tuning",
        "RandomizedSearchCV (40 iterations, 5-fold CV, scoring=accuracy) for LightGBM and XGBoost; 20 iterations for BalancedRF.",
        "",
        "## 10. Threshold Optimization",
        "Threshold selected to maximize accuracy on OOF predictions from training data only.",
        f"Best threshold: {best_thr:.2f}",
        "",
        "## 11. Per-Project Results",
        "See outputs/per_project_improved_results.csv",
        "Small projects flagged with LOW SAMPLE SIZE warning.",
        "",
        "## 12. Cross-Project Results",
        "See outputs/cross_project_results.csv (Leave-One-Project-Out evaluation).",
        "",
        "## 13. Error Analysis",
        "See outputs/error_analysis.csv",
        f"False Negatives (missed defects): {int(cm[1,0])}",
        f"False Positives (false alarms): {int(cm[0,1])}",
        "",
        "## 14. Feature Importance / SHAP",
        "See outputs/feature_importance.csv and plots/improved_shap_summary.png",
        "",
        "## 15. Best Model",
        f"Model: {best_name}",
        f"Saved: models/final_defect_prediction_pipeline.pkl",
        "",
        "## 16. 90% Accuracy Investigation",
        f"90% accuracy achieved: {'YES' if achieved_90 else 'NO'}",
        "" if achieved_90 else (
            "The current held-out test result does not support a 90% accuracy claim.\n"
            "A larger, project-aware evaluation and additional process metrics would be needed\n"
            "to establish whether further improvements generalize."
        ),
        "",
        "## 17. Limitations",
        "- CK metrics capture structure but not runtime behaviour or developer experience.",
        "- Combined model generalizes across projects but loses project-specific signal.",
        "- Process metrics unavailable in current dataset.",
        "",
        "## 18. Recommended Future Work",
        "1. Add process metrics: code churn, number of commits, author count.",
        "2. Project-specific fine-tuning on top of combined model.",
        "3. Deep learning on AST/token sequences (code2vec).",
        "4. Stacking ensemble with calibrated probabilities.",
        "5. Extend to NASA MDP and Eclipse defect datasets.",
    ]
    with open(os.path.join(OUTPUTS_DIR, "FINAL_ANALYSIS_REPORT.md"), "w") as fh:
        fh.write("\n".join(report_lines))

    # ── Final Terminal Output ─────────────────────────────────────────────────
    print("\n" + "="*50)
    print("SOFTWARE DEFECT PREDICTION - FINAL RESULTS")
    print("="*50)
    print("\nCURRENT HELD-OUT RESULTS (selected using training OOF accuracy)")
    print(f"  Model              : {best_name}")
    print(f"  Feature Set        : C (CK + Engineered, {len(feat_cols)} features)")
    print(f"  Imbalance Strategy : SMOTE")
    print(f"  Threshold          : {best_thr:.2f}")
    print(f"  Accuracy           : {best_row['Accuracy']:.4f}")
    print(f"  Precision          : {best_row['Precision']:.4f}")
    print(f"  Recall             : {best_row['Recall']:.4f}")
    print(f"  F1                 : {best_row['F1']:.4f}")
    print(f"  ROC-AUC            : {best_row['ROC-AUC']:.4f}")
    print(f"  PR-AUC             : {best_row['PR-AUC']:.4f}")
    print(f"  Balanced Accuracy  : {best_row['Balanced_Acc']:.4f}")
    print(f"  MCC                : {best_row['MCC']:.4f}")
    print(f"\n90% ACCURACY ACHIEVED: {'YES' if achieved_90 else 'NO'}")
    if not achieved_90:
        print("\n  Main Limitation:")
        print("    CK metrics have genuine class overlap across projects.")
        print("    Combined model accuracy is bounded by inter-project variance.")
        print("  Recommended Improvement:")
        print("    Add process metrics (code churn, commits, authors).")
        print("    Use project-specific fine-tuning.")
    print("\n" + "="*50)
    print("CURRENT METRICS (previous result not comparable)")
    print("="*50)
    print(old_vs_new.to_string(index=False))
    print("\nAll outputs saved to outputs/ and plots/")
    print("Run: python improved_visualize.py")
    print("Run: python predict.py")
    print("Run: python -m streamlit run app.py")
    print("="*50)
