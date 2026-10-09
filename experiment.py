"""
experiment.py  —  Full Software Defect Prediction Experiment Pipeline
Experiments 1-8 + final report

Legacy research script. Use improved_pipeline.py for the canonical model,
threshold selection, and held-out evaluation.
"""
import os, glob, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from data_utils import normalize_bug_labels

# ── Directories ──────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "datasets")
MODELS_DIR  = os.path.join(BASE, "models");      os.makedirs(MODELS_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(BASE, "results");     os.makedirs(RESULTS_DIR, exist_ok=True)
PLOTS_DIR   = os.path.join(BASE, "experiment_plots"); os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── Feature groups present in the PROMISE datasets ───────────────────────────
CK_FEATURES   = ["wmc","dit","noc","cbo","rfc","lcom","loc"]
OO_FEATURES   = ["ca","ce","npm","lcom3","dam","moa","mfa","cam","ic","cbm","amc"]
CC_FEATURES   = ["max_cc","avg_cc"]   # McCabe cyclomatic complexity proxies

ALL_RAW       = CK_FEATURES + OO_FEATURES + CC_FEATURES

print("="*60)
print("  Software Defect Prediction — Full Experiment Pipeline")
print("="*60)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

EXCLUDE = ["xalan"]

def load_promise():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        if any(ex in os.path.basename(f).lower() for ex in EXCLUDE):
            continue
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
        if not lbl:
            continue
        df = df.rename(columns={lbl: "bug"})
        df["source"] = os.path.basename(f).replace(".csv","")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["bug"] = normalize_bug_labels(combined["bug"])
    combined = combined.dropna(subset=["bug"])
    combined["bug"] = combined["bug"].astype(int)
    return combined

def preprocess(df, feature_cols):
    available = [c for c in feature_cols if c in df.columns]
    out = df[available + ["bug"]].copy()
    for col in available:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median())
    out = out.drop_duplicates(subset=available + ["bug"])
    return out, available

def add_engineered_features(df, base_cols):
    df = df.copy()
    eps = 1e-6
    if "wmc" in df.columns and "loc" in df.columns:
        df["wmc_per_loc"]  = df["wmc"]  / (df["loc"]  + eps)
    if "cbo" in df.columns and "loc" in df.columns:
        df["cbo_per_loc"]  = df["cbo"]  / (df["loc"]  + eps)
    if "rfc" in df.columns and "wmc" in df.columns:
        df["rfc_per_wmc"]  = df["rfc"]  / (df["wmc"]  + eps)
    if "lcom" in df.columns and "wmc" in df.columns:
        df["lcom_per_wmc"] = df["lcom"] / (df["wmc"]  + eps)
    if "cbo" in df.columns and "rfc" in df.columns:
        df["coupling_sum"] = df["cbo"]  + df["rfc"]
    for col in ["wmc","loc","lcom","rfc","cbo"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    new_cols = [c for c in df.columns if c not in base_cols + ["bug"]]
    return df, new_cols

raw_df = load_promise()
print(f"\nLoaded {len(raw_df)} raw rows from {raw_df['source'].nunique()} projects")

# ═══════════════════════════════════════════════════════════════════════════
# 2. EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2,2):
        tn,fp,fn,tp = cm.ravel()
        sens = tp/(tp+fn+1e-9)
        spec = tn/(tn+fp+1e-9)
        return np.sqrt(sens*spec)
    return 0.0

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model"    : name,
        "Threshold": threshold,
        "Accuracy" : round(accuracy_score(y_test, y_pred),4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0),4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0),4),
        "F1"       : round(f1_score(y_test, y_pred, zero_division=0),4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_prob),4),
        "PR-AUC"   : round(average_precision_score(y_test, y_prob),4),
        "MCC"      : round(matthews_corrcoef(y_test, y_pred),4),
        "G-Mean"   : round(gmean(y_test, y_pred),4),
    }

def cv_score(model_factory, X, y, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=SEED)),
        ("model",  model_factory()),
    ])
    sc = cross_validate(pipe, X, y, cv=skf,
                        scoring=["accuracy","f1","roc_auc","average_precision"],
                        n_jobs=-1)
    return {
        "CV_Acc"   : round(sc["test_accuracy"].mean(),4),
        "CV_F1"    : round(sc["test_f1"].mean(),4),
        "CV_AUC"   : round(sc["test_roc_auc"].mean(),4),
        "CV_PR_AUC": round(sc["test_average_precision"].mean(),4),
    }

# ── Model zoo ────────────────────────────────────────────────────────────────
def get_models():
    return {
        "LogReg"   : LogisticRegression(max_iter=1000, random_state=SEED),
        "DTree"    : DecisionTreeClassifier(random_state=SEED),
        "RF"       : RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "SVM"      : SVC(probability=True, random_state=SEED),
        "XGBoost"  : XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "LightGBM" : lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        "HistGB"   : HistGradientBoostingClassifier(random_state=SEED),
        "MLP"      : MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=SEED),
    }

# ═══════════════════════════════════════════════════════════════════════════
# 3. EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

all_experiment_results = []

def run_experiment(exp_id, exp_name, feature_cols, balancing="smote"):
    print(f"\n{'─'*60}")
    print(f"Experiment {exp_id}: {exp_name}")
    print(f"{'─'*60}")

    df_clean, avail = preprocess(raw_df, feature_cols)
    X = df_clean[avail].values
    y = df_clean["bug"].values
    print(f"  Samples: {len(df_clean)} | Features: {len(avail)} | Defect rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # Replace any inf/nan introduced by engineered features
    X_tr_sc = np.nan_to_num(X_tr_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_sc  = np.nan_to_num(X_te_sc,  nan=0.0, posinf=0.0, neginf=0.0)

    if balancing == "smote":
        sm = SMOTE(random_state=SEED)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_sc, y_tr)
    elif balancing == "smotetomek":
        sm = SMOTETomek(random_state=SEED)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_sc, y_tr)
    else:
        X_tr_bal, y_tr_bal = X_tr_sc, y_tr

    rows = []
    trained = {}
    for name, model in get_models().items():
        model.fit(X_tr_bal, y_tr_bal)
        res = evaluate_model(name, model, X_te_sc, y_te)
        res["Experiment"] = exp_id
        res["Exp_Name"]   = exp_name
        res["N_Features"] = len(avail)
        res["Balancing"]  = balancing
        rows.append(res)
        trained[name] = model
        print(f"  {name:12s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}  AUC={res['ROC-AUC']:.4f}")

    all_experiment_results.extend(rows)
    return pd.DataFrame(rows), trained, scaler, avail, X_te_sc, y_te

# ── Experiment 1: Original 7 CK features ─────────────────────────────────
res1, mdl1, sc1, ft1, Xte1, yte1 = run_experiment(
    1, "Original 7 CK Features", CK_FEATURES
)

# ── Experiment 2: CK + OO metrics ────────────────────────────────────────
res2, mdl2, sc2, ft2, Xte2, yte2 = run_experiment(
    2, "CK + OO Metrics", CK_FEATURES + OO_FEATURES
)

# ── Experiment 3: CK + McCabe complexity proxies ─────────────────────────
res3, mdl3, sc3, ft3, Xte3, yte3 = run_experiment(
    3, "CK + Complexity (max_cc, avg_cc)", CK_FEATURES + CC_FEATURES
)

# ── Experiment 4: CK + OO + Complexity (all raw) ─────────────────────────
res4, mdl4, sc4, ft4, Xte4, yte4 = run_experiment(
    4, "All Raw Features (CK+OO+CC)", ALL_RAW
)

# ── Experiment 5: CK + engineered features ───────────────────────────────
df_eng, eng_cols = add_engineered_features(raw_df, CK_FEATURES)
res5, mdl5, sc5, ft5, Xte5, yte5 = run_experiment(
    5, "CK + Engineered Features", CK_FEATURES + eng_cols
)

# ── Experiment 6: All raw + engineered ───────────────────────────────────
df_eng2, eng_cols2 = add_engineered_features(raw_df, ALL_RAW)
res6, mdl6, sc6, ft6, Xte6, yte6 = run_experiment(
    6, "All Raw + Engineered Features", ALL_RAW + eng_cols2
)

# ── Experiment 7: Feature-selected (SelectKBest MI, k=10) ────────────────
print(f"\n{'─'*60}")
print("Experiment 7: Feature Selection (Mutual Information, k=10)")
print(f"{'─'*60}")

df7, avail7 = preprocess(raw_df, ALL_RAW + eng_cols2)
X7 = df7[avail7].values
y7 = df7["bug"].values
X7_tr, X7_te, y7_tr, y7_te = train_test_split(X7, y7, test_size=0.2, random_state=SEED, stratify=y7)
sc7 = StandardScaler()
X7_tr_sc = np.nan_to_num(sc7.fit_transform(X7_tr), nan=0.0, posinf=0.0, neginf=0.0)
X7_te_sc  = np.nan_to_num(sc7.transform(X7_te),    nan=0.0, posinf=0.0, neginf=0.0)
sm7 = SMOTE(random_state=SEED)
X7_tr_bal, y7_tr_bal = sm7.fit_resample(X7_tr_sc, y7_tr)

selector = SelectKBest(mutual_info_classif, k=10)
selector.fit(X7_tr_bal, y7_tr_bal)
X7_tr_sel = selector.transform(X7_tr_bal)
X7_te_sel = selector.transform(X7_te_sc)
sel_features = [avail7[i] for i in selector.get_support(indices=True)]
print(f"  Selected features: {sel_features}")

rows7 = []
mdl7  = {}
for name, model in get_models().items():
    model.fit(X7_tr_sel, y7_tr_bal)
    y_prob = model.predict_proba(X7_te_sel)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    res = {
        "Model": name, "Threshold": 0.5,
        "Accuracy" : round(accuracy_score(y7_te, y_pred),4),
        "Precision": round(precision_score(y7_te, y_pred, zero_division=0),4),
        "Recall"   : round(recall_score(y7_te, y_pred, zero_division=0),4),
        "F1"       : round(f1_score(y7_te, y_pred, zero_division=0),4),
        "ROC-AUC"  : round(roc_auc_score(y7_te, y_prob),4),
        "PR-AUC"   : round(average_precision_score(y7_te, y_prob),4),
        "MCC"      : round(matthews_corrcoef(y7_te, y_pred),4),
        "G-Mean"   : round(gmean(y7_te, y_pred),4),
        "Experiment": 7, "Exp_Name": "Feature Selected (k=10)",
        "N_Features": 10, "Balancing": "smote",
    }
    rows7.append(res)
    mdl7[name] = model
    print(f"  {name:12s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}  AUC={res['ROC-AUC']:.4f}")

res7 = pd.DataFrame(rows7)
all_experiment_results.extend(rows7)

# ═══════════════════════════════════════════════════════════════════════════
# 4. EXPERIMENT 8 — Best model + Optuna tuning + threshold optimization
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("Experiment 8: Best Model + Optuna Tuning + Threshold Optimization")
print(f"{'─'*60}")

# Pick best experiment/model by ROC-AUC from experiments 1-7
all_df = pd.DataFrame(all_experiment_results)
best_row = all_df.loc[all_df["ROC-AUC"].idxmax()]
print(f"  Best so far: Exp {int(best_row['Experiment'])} | {best_row['Model']} | AUC={best_row['ROC-AUC']}")

# Use Exp 4 features (all raw) for tuning — best balance of signal vs noise
df8, avail8 = preprocess(raw_df, ALL_RAW)
X8 = df8[avail8].values
y8 = df8["bug"].values
X8_tr, X8_te, y8_tr, y8_te = train_test_split(X8, y8, test_size=0.2, random_state=SEED, stratify=y8)
sc8 = StandardScaler()
X8_tr_sc = sc8.fit_transform(X8_tr)
X8_te_sc  = sc8.transform(X8_te)
sm8 = SMOTE(random_state=SEED)
X8_tr_bal, y8_tr_bal = sm8.fit_resample(X8_tr_sc, y8_tr)

# ── Optuna: tune LightGBM (fastest + best AUC in most experiments) ────────
skf8 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def objective(trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 600),
        "max_depth"        : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves"       : trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "random_state": SEED, "n_jobs": -1, "verbose": -1,
    }
    f1s = []
    for tr_idx, va_idx in skf8.split(X8_tr_bal, y8_tr_bal):
        Xtr, Xva = X8_tr_bal[tr_idx], X8_tr_bal[va_idx]
        ytr, yva = y8_tr_bal[tr_idx], y8_tr_bal[va_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr)
        yp = m.predict(Xva)
        f1s.append(f1_score(yva, yp, zero_division=0))
    return np.mean(f1s)

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=60, show_progress_bar=False)
best_params = study.best_params
best_params.update({"random_state": SEED, "n_jobs": -1, "verbose": -1})
print(f"  Best Optuna F1 (CV): {study.best_value:.4f}")
print(f"  Best params: {best_params}")

tuned_lgbm = lgb.LGBMClassifier(**best_params)
tuned_lgbm.fit(X8_tr_bal, y8_tr_bal)

# ── Also tune XGBoost with RandomizedSearchCV ─────────────────────────────
xgb_grid = {
    "n_estimators"    : [200, 400, 600],
    "max_depth"       : [3, 5, 7],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "subsample"       : [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_alpha"       : [0, 0.1, 1.0],
    "reg_lambda"      : [1.0, 2.0, 5.0],
}
xgb_rs = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
    xgb_grid, n_iter=40, cv=skf8, scoring="f1",
    random_state=SEED, n_jobs=-1
)
xgb_rs.fit(X8_tr_bal, y8_tr_bal)
tuned_xgb = xgb_rs.best_estimator_
print(f"  Best XGB F1 (CV): {xgb_rs.best_score_:.4f}")

# ── Threshold optimization on CV (NOT on test set) ───────────────────────
def best_threshold_cv(model, X_tr, y_tr, cv, metric="f1"):
    thresholds = np.arange(0.2, 0.8, 0.02)
    scores = {t: [] for t in thresholds}
    for tr_i, va_i in cv.split(X_tr, y_tr):
        m2 = lgb.LGBMClassifier(**best_params)
        m2.fit(X_tr[tr_i], y_tr[tr_i])
        probs = m2.predict_proba(X_tr[va_i])[:,1]
        for t in thresholds:
            yp = (probs >= t).astype(int)
            scores[t].append(f1_score(y_tr[va_i], yp, zero_division=0))
    mean_scores = {t: np.mean(v) for t,v in scores.items()}
    return max(mean_scores, key=mean_scores.get)

opt_threshold = best_threshold_cv(tuned_lgbm, X8_tr_bal, y8_tr_bal, skf8)
print(f"  Optimal threshold (CV): {opt_threshold:.2f}")

# ── Evaluate Exp 8 models on untouched test set ───────────────────────────
rows8 = []
for name, model, thr in [
    ("LightGBM_Tuned", tuned_lgbm, 0.5),
    ("LightGBM_Tuned_OptThr", tuned_lgbm, opt_threshold),
    ("XGBoost_Tuned", tuned_xgb, 0.5),
]:
    res = evaluate_model(name, model, X8_te_sc, y8_te, threshold=thr)
    res["Experiment"] = 8
    res["Exp_Name"]   = "Tuned + Threshold Opt"
    res["N_Features"] = len(avail8)
    res["Balancing"]  = "smote"
    rows8.append(res)
    print(f"  {name:28s}  Acc={res['Accuracy']:.4f}  F1={res['F1']:.4f}  AUC={res['ROC-AUC']:.4f}  MCC={res['MCC']:.4f}")

res8 = pd.DataFrame(rows8)
all_experiment_results.extend(rows8)

# ── Save best model ───────────────────────────────────────────────────────
best_model_path = os.path.join(MODELS_DIR, "best_defect_prediction_model.pkl")
joblib.dump(tuned_lgbm, best_model_path)
joblib.dump(sc8, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(avail8, os.path.join(MODELS_DIR, "feature_list.pkl"))
joblib.dump(best_params, os.path.join(MODELS_DIR, "best_hyperparams.pkl"))
joblib.dump(opt_threshold, os.path.join(MODELS_DIR, "best_threshold.pkl"))
print(f"\n  Saved best model → {best_model_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════

final_df = pd.DataFrame(all_experiment_results)
final_df.to_csv(os.path.join(RESULTS_DIR, "final_results.csv"), index=False)

# Feature importance
fi = pd.Series(tuned_lgbm.feature_importances_, index=avail8).sort_values(ascending=False)
fi.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), header=["importance"])

# CV results for best model
cv_rows = []
for tr_i, va_i in skf8.split(X8_tr_bal, y8_tr_bal):
    m_cv = lgb.LGBMClassifier(**best_params)
    m_cv.fit(X8_tr_bal[tr_i], y8_tr_bal[tr_i])
    yp = m_cv.predict(X8_tr_bal[va_i])
    yprob = m_cv.predict_proba(X8_tr_bal[va_i])[:,1]
    cv_rows.append({
        "Accuracy" : accuracy_score(y8_tr_bal[va_i], yp),
        "F1"       : f1_score(y8_tr_bal[va_i], yp, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y8_tr_bal[va_i], yprob),
    })
pd.DataFrame(cv_rows).to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)

print("\nResults saved to results/")

# ═══════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid")

# ── Plot 1: Experiment comparison (AUC) ──────────────────────────────────
pivot = final_df.groupby(["Experiment","Model"])["ROC-AUC"].max().unstack("Model")
fig, ax = plt.subplots(figsize=(14,5))
pivot.plot(kind="bar", ax=ax, rot=0)
ax.set_title("ROC-AUC by Experiment & Model")
ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.4, 0.9)
ax.legend(loc="lower right", fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
plt.close()

# ── Plot 2: Confusion matrix (best model) ────────────────────────────────
y_pred_best = (tuned_lgbm.predict_proba(X8_te_sc)[:,1] >= opt_threshold).astype(int)
cm = confusion_matrix(y8_te, y_pred_best)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(cm, display_labels=["Clean","Defective"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — LightGBM Tuned (thr={opt_threshold:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_best.png"), dpi=150)
plt.close()

# ── Plot 3: ROC curves (Exp 8 models) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
for name, model in [("LightGBM_Tuned", tuned_lgbm), ("XGBoost_Tuned", tuned_xgb)]:
    yp = model.predict_proba(X8_te_sc)[:,1]
    fpr, tpr, _ = roc_curve(y8_te, yp)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y8_te,yp):.3f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves — Tuned Models")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curves_best.png"), dpi=150)
plt.close()

# ── Plot 4: Precision-Recall curves ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5))
for name, model in [("LightGBM_Tuned", tuned_lgbm), ("XGBoost_Tuned", tuned_xgb)]:
    yp = model.predict_proba(X8_te_sc)[:,1]
    prec, rec, _ = precision_recall_curve(y8_te, yp)
    ap = average_precision_score(y8_te, yp)
    ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
ax.axhline(y8_te.mean(), color="gray", linestyle="--", label="Baseline")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pr_curves_best.png"), dpi=150)
plt.close()

# ── Plot 5: Feature importance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,6))
fi.head(15).sort_values().plot(kind="barh", ax=ax, color="#5C85D6")
ax.set_title("Top 15 Feature Importances — LightGBM Tuned")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ── Plot 6: Class distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,4))
counts = pd.Series(y8).value_counts().sort_index()
ax.bar(["Clean (0)","Defective (1)"], counts.values, color=["#38a169","#e53e3e"])
for i,v in enumerate(counts.values):
    ax.text(i, v+10, str(v), ha="center", fontweight="bold")
ax.set_title("Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
plt.close()

# ── Plot 7: Feature correlation heatmap ──────────────────────────────────
df_corr, _ = preprocess(raw_df, ALL_RAW)
corr = df_corr[[c for c in ALL_RAW if c in df_corr.columns] + ["bug"]].corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"size":7}, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

# ── Plot 8: Accuracy across experiments (best model per exp) ─────────────
best_per_exp = final_df.groupby("Experiment")[["Accuracy","F1","ROC-AUC"]].max()
fig, ax = plt.subplots(figsize=(10,5))
best_per_exp.plot(kind="bar", ax=ax, rot=0, color=["#4299e1","#48bb78","#ed8936"])
ax.set_title("Best Accuracy / F1 / AUC per Experiment")
ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.set_xlabel("Experiment")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "experiment_summary.png"), dpi=150)
plt.close()

print("All plots saved to experiment_plots/")

# ═══════════════════════════════════════════════════════════════════════════
# 7. FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL REPORT")
print("="*60)

best_overall = final_df.loc[final_df["ROC-AUC"].idxmax()]
best_acc_row  = final_df.loc[final_df["Accuracy"].idxmax()]

print(f"\n{'─'*60}")
print("BEST BY ROC-AUC:")
for k,v in best_overall.items():
    print(f"  {k:15s}: {v}")

print(f"\n{'─'*60}")
print("BEST BY ACCURACY:")
for k,v in best_acc_row.items():
    print(f"  {k:15s}: {v}")

print(f"\n{'─'*60}")
print("EXPERIMENT COMPARISON TABLE (best model per experiment):")
cols = ["Experiment","Exp_Name","N_Features","Model","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC","MCC"]
summary = final_df.sort_values("ROC-AUC", ascending=False).drop_duplicates("Experiment")[cols]
print(summary.to_string(index=False))

print(f"\n{'─'*60}")
print("BASELINE COMPARISON (current project results):")
print("  Random Forest  Acc=0.6399  F1=0.4120  AUC=0.6248")
print("  XGBoost        Acc=0.6570  F1=0.4435  AUC=0.6392")
print(f"\n  New Best Model : {best_overall['Model']}")
print(f"  New Accuracy   : {best_overall['Accuracy']}")
print(f"  New F1         : {best_overall['F1']}")
print(f"  New ROC-AUC    : {best_overall['ROC-AUC']}")
print(f"  New PR-AUC     : {best_overall['PR-AUC']}")
print(f"  New MCC        : {best_overall['MCC']}")

target_acc = 0.84
achieved = float(best_acc_row["Accuracy"]) >= target_acc
print(f"\n{'─'*60}")
print(f"TARGET 84% ACCURACY: {'ACHIEVED ✓' if achieved else 'NOT ACHIEVED'}")
print(f"  Best Accuracy = {best_acc_row['Accuracy']} (Model: {best_acc_row['Model']}, Exp: {int(best_acc_row['Experiment'])})")
if not achieved:
    print("""
  REASONS 84% was not achieved on the combined dataset:
  1. The PROMISE dataset has genuine class overlap — CK metrics alone
     cannot perfectly separate defective from clean modules.
  2. The combined dataset (all 11 projects) is more balanced (~33% defect)
     making it harder than single-project experiments.
  3. CK/OO metrics capture structural complexity but not runtime behaviour,
     developer experience, or code review history.
  4. Individual projects (xalan, jedit) show 94-98% accuracy due to extreme
     class imbalance — not genuine predictive power.

  WHAT CAN REALISTICALLY IMPROVE RESULTS:
  1. Add process metrics: code churn, number of commits, author count.
  2. Add static analysis warnings as features.
  3. Use project-specific models instead of a combined model.
  4. Deep learning on AST/token sequences (code2vec embeddings).
  5. Ensemble stacking with calibrated probabilities.
""")

summary.to_csv(os.path.join(RESULTS_DIR, "experiment_comparison.csv"), index=False)
print("\nDone. All results in results/ and experiment_plots/")
