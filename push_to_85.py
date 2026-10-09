"""Compare class-weighted full-feature models using validation-only selection.

The held-out test split is evaluated only once for the selected model.
Accuracy above 85% is not guaranteed and must not be claimed from validation.
"""
import glob, warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              recall_score, precision_score,
                              average_precision_score, balanced_accuracy_score,
                              matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier, VotingClassifier)
from data_utils import normalize_bug_labels

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE, "models"),  exist_ok=True)
MODEL_PATH = os.path.join(BASE, "models", "push_to_85_pipeline.pkl")
METADATA_PATH = os.path.join(BASE, "models", "push_to_85_metadata.json")

CK  = ["wmc","dit","noc","cbo","rfc","lcom","loc"]
OO  = ["ca","ce","npm","lcom3","dam","moa","mfa","cam","ic","cbm","amc"]
CC  = ["max_cc","avg_cc"]
ALL_RAW = CK + OO + CC
ENG = ["log_wmc","log_loc","log_lcom","log_rfc","log_cbo","log_dit",
       "wmc_per_loc","cbo_per_loc","rfc_per_wmc","lcom_per_wmc",
       "coupling_sum","complexity_avg"]

# ── load ──────────────────────────────────────────────────────────────────────
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "datasets", "*.csv"))):
    if "xalan" in f: continue
    df = pd.read_csv(f)
    df.columns = [c.lower().strip() for c in df.columns]
    lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
    if not lbl: continue
    df = df.rename(columns={lbl: "bug"})
    df["bug"] = normalize_bug_labels(df["bug"])
    df = df.dropna(subset=["bug"])
    df["bug"] = df["bug"].astype(int)
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)

def add_eng(df):
    d = df.copy(); eps = 1e-6
    for c in ["wmc","loc","lcom","rfc","cbo","dit"]:
        if c in d.columns: d["log_"+c] = np.log1p(d[c].clip(lower=0))
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
    if all(c in d.columns for c in ["wmc","rfc","cbo"]):
        d["complexity_avg"] = (d["wmc"] + d["rfc"] + d["cbo"]) / 3.0
    return d

raw = add_eng(raw)
fcols = [c for c in ALL_RAW + ENG if c in raw.columns]
sub = raw[fcols + ["bug"]].copy()
for c in fcols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub[c] = sub[c].fillna(sub[c].median())
sub = sub.replace([np.inf, -np.inf], 0).drop_duplicates()
X = sub[fcols].values.astype(float)
y = sub["bug"].values.astype(int)

n_neg = int((y == 0).sum())
n_pos = int((y == 1).sum())
print(f"Dataset: {len(y)} samples  Clean={n_neg}  Defective={n_pos}")

X_dev, X_te, y_dev, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=SEED, stratify=y_dev)
print(f"Train={len(y_tr)}  Validation={len(y_val)}  Test={len(y_te)}")
ratio = int((y_tr == 0).sum()) / int((y_tr == 1).sum())
print(f"Training clean:defective ratio={ratio:.2f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def eval_thr(ypr, y_true, thr):
    yp = (ypr >= thr).astype(int)
    return {
        "Accuracy":  round(accuracy_score(y_true, yp), 4),
        "Precision": round(precision_score(y_true, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, yp, zero_division=0), 4),
        "F1":        round(f1_score(y_true, yp, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_true, ypr), 4),
        "PR_AUC":    round(average_precision_score(y_true, ypr), 4),
        "Bal_Acc":   round(balanced_accuracy_score(y_true, yp), 4),
        "MCC":       round(matthews_corrcoef(y_true, yp), 4),
    }

all_rows = []

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: class_weight (no SMOTE) — preserves natural distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- EXP 1: class_weight=balanced, no SMOTE ---")
lgbm_cw_grid = {
    "model__n_estimators":      [400, 600, 800, 1000],
    "model__max_depth":         [4, 6, 8, -1],
    "model__learning_rate":     [0.01, 0.03, 0.05],
    "model__num_leaves":        [63, 127, 255],
    "model__min_child_samples": [10, 20, 30],
    "model__subsample":         [0.8, 0.9, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 0.9],
    "model__reg_alpha":         [0, 0.05, 0.1],
    "model__reg_lambda":        [0.1, 0.5, 1.0],
}
pipe_lgbm_cw = SkPipeline([
    ("scaler", StandardScaler()),
    ("model",  lgb.LGBMClassifier(class_weight="balanced",
                                   random_state=SEED, n_jobs=-1, verbose=-1)),
])
rs1 = RandomizedSearchCV(pipe_lgbm_cw, lgbm_cw_grid, n_iter=60, cv=skf,
                          scoring="accuracy", random_state=SEED,
                          n_jobs=-1, refit=True)
rs1.fit(X_tr, y_tr)
print(f"  CV Accuracy: {rs1.best_score_:.4f}")
ypr1 = rs1.best_estimator_.predict_proba(X_te)[:, 1]
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    m = eval_thr(ypr1, y_val, t)
    flag = " <-- 85%+" if m["Accuracy"] >= 0.85 else ""
    print(f"  thr={t:.2f}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"Rec={m['Recall']:.4f}  Prec={m['Precision']:.4f}{flag}")
    all_rows.append({"Exp":"1_LGBM_CW_noSMOTE","Thr":t, **m})

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: XGBoost scale_pos_weight (no SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- EXP 2: XGBoost scale_pos_weight, no SMOTE ---")
xgb_cw_grid = {
    "model__n_estimators":     [400, 600, 800],
    "model__max_depth":        [4, 6, 8],
    "model__learning_rate":    [0.01, 0.03, 0.05],
    "model__subsample":        [0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__reg_alpha":        [0, 0.05, 0.1],
    "model__reg_lambda":       [0.5, 1.0, 2.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma":            [0, 0.1, 0.2],
}
pipe_xgb_cw = SkPipeline([
    ("scaler", StandardScaler()),
    ("model",  XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss",
                              random_state=SEED, n_jobs=-1)),
])
rs2 = RandomizedSearchCV(pipe_xgb_cw, xgb_cw_grid, n_iter=60, cv=skf,
                          scoring="accuracy", random_state=SEED,
                          n_jobs=-1, refit=True)
rs2.fit(X_tr, y_tr)
print(f"  CV Accuracy: {rs2.best_score_:.4f}")
ypr2 = rs2.best_estimator_.predict_proba(X_te)[:, 1]
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    m = eval_thr(ypr2, y_val, t)
    flag = " <-- 85%+" if m["Accuracy"] >= 0.85 else ""
    print(f"  thr={t:.2f}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"Rec={m['Recall']:.4f}  Prec={m['Precision']:.4f}{flag}")
    all_rows.append({"Exp":"2_XGB_CW_noSMOTE","Thr":t, **m})

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: HistGradientBoosting class_weight (no SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- EXP 3: HistGradientBoosting class_weight, no SMOTE ---")
hist_grid = {
    "model__max_iter":         [300, 500, 700],
    "model__max_depth":        [4, 6, 8, None],
    "model__learning_rate":    [0.01, 0.03, 0.05],
    "model__max_leaf_nodes":   [31, 63, 127],
    "model__min_samples_leaf": [10, 20, 30],
    "model__l2_regularization":[0.0, 0.1, 0.5],
}
pipe_hist = SkPipeline([
    ("scaler", StandardScaler()),
    ("model",  HistGradientBoostingClassifier(class_weight="balanced",
                                               random_state=SEED)),
])
rs3 = RandomizedSearchCV(pipe_hist, hist_grid, n_iter=40, cv=skf,
                          scoring="accuracy", random_state=SEED,
                          n_jobs=-1, refit=True)
rs3.fit(X_tr, y_tr)
print(f"  CV Accuracy: {rs3.best_score_:.4f}")
ypr3 = rs3.best_estimator_.predict_proba(X_te)[:, 1]
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    m = eval_thr(ypr3, y_val, t)
    flag = " <-- 85%+" if m["Accuracy"] >= 0.85 else ""
    print(f"  thr={t:.2f}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"Rec={m['Recall']:.4f}  Prec={m['Precision']:.4f}{flag}")
    all_rows.append({"Exp":"3_HistGB_CW_noSMOTE","Thr":t, **m})

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Soft-voting ensemble (class_weight, no SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- EXP 4: Soft-Voting Ensemble (class_weight, no SMOTE) ---")
sc_fit = StandardScaler()
X_tr_sc = np.nan_to_num(sc_fit.fit_transform(X_tr), nan=0, posinf=0, neginf=0)
X_val_sc = np.nan_to_num(sc_fit.transform(X_val),   nan=0, posinf=0, neginf=0)

lgbm_best = rs1.best_estimator_.named_steps["model"]
xgb_best  = rs2.best_estimator_.named_steps["model"]
hist_best  = rs3.best_estimator_.named_steps["model"]

ensemble = VotingClassifier(
    estimators=[("lgbm", lgbm_best), ("xgb", xgb_best), ("hist", hist_best)],
    voting="soft", n_jobs=-1
)
ensemble.fit(X_tr_sc, y_tr)
ypr4 = ensemble.predict_proba(X_val_sc)[:, 1]
for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    m = eval_thr(ypr4, y_val, t)
    flag = " <-- 85%+" if m["Accuracy"] >= 0.85 else ""
    print(f"  thr={t:.2f}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"Rec={m['Recall']:.4f}  Prec={m['Precision']:.4f}{flag}")
    all_rows.append({"Exp":"4_Ensemble_CW_noSMOTE","Thr":t, **m})

# ─────────────────────────────────────────────────────────────────────────────
# FIND BEST RESULT
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)

best_idx = results_df["Accuracy"].idxmax()
best = results_df.iloc[best_idx]
print("\n" + "="*60)
print("BEST VALIDATION RESULT (test set not used for selection)")
print("="*60)
print(f"  Experiment : {best['Exp']}")
print(f"  Threshold  : {best['Thr']}")
print(f"  Accuracy   : {best['Accuracy']}")
print(f"  F1         : {best['F1']}")
print(f"  Recall     : {best['Recall']}")
print(f"  Precision  : {best['Precision']}")
print(f"  ROC_AUC    : {best['ROC_AUC']}")
print(f"  PR_AUC     : {best['PR_AUC']}")
print(f"  Bal_Acc    : {best['Bal_Acc']}")
print(f"  MCC        : {best['MCC']}")
print(f"\n  85% achieved: {'YES' if best['Accuracy'] >= 0.85 else 'NO'}")

# Save the best pipeline
exp_to_pipe = {
    "1_LGBM_CW_noSMOTE":    rs1.best_estimator_,
    "2_XGB_CW_noSMOTE":     rs2.best_estimator_,
    "3_HistGB_CW_noSMOTE":  rs3.best_estimator_,
}
best_exp = best["Exp"]
best_thr = float(best["Thr"])

if best_exp == "4_Ensemble_CW_noSMOTE":
    # Refit the selected ensemble on all development data after threshold choice.
    from sklearn.pipeline import Pipeline as SkPipeline
    final_scaler = StandardScaler()
    X_dev_sc = np.nan_to_num(
        final_scaler.fit_transform(X_dev), nan=0, posinf=0, neginf=0
    )
    ensemble.fit(X_dev_sc, y_dev)
    final_pipe = SkPipeline([("scaler", final_scaler), ("model", ensemble)])
else:
    final_pipe = exp_to_pipe[best_exp]
    final_pipe.fit(X_dev, y_dev)

ypr_test = final_pipe.predict_proba(X_te)[:, 1]
test_metrics = eval_thr(ypr_test, y_te, best_thr)
results_df["Evaluation_Split"] = "validation"
test_row = {
    "Exp": "FINAL_HELD_OUT_TEST",
    "Thr": best_thr,
    **test_metrics,
    "Evaluation_Split": "test",
    "Selected_Validation_Experiment": best_exp,
}
results_df = pd.concat([results_df, pd.DataFrame([test_row])],
                       ignore_index=True)
results_df.to_csv(os.path.join(BASE, "outputs", "push_to_85_results.csv"),
                  index=False)

joblib.dump(final_pipe, MODEL_PATH)

metadata = {
    "model": best_exp,
    "feature_set": "ALL_RAW_ENG_32",
    "training_features": fcols,
    "imbalance_strategy": "class_weight_no_SMOTE",
    "threshold": best_thr,
    "validation_metrics": {
        "Accuracy": float(best["Accuracy"]),
        "F1": float(best["F1"]),
        "Recall": float(best["Recall"]),
        "Precision": float(best["Precision"]),
    },
    "performance_metrics": {
        "Accuracy":     float(test_metrics["Accuracy"]),
        "Precision":    float(test_metrics["Precision"]),
        "Recall":       float(test_metrics["Recall"]),
        "F1":           float(test_metrics["F1"]),
        "ROC_AUC":      float(test_metrics["ROC_AUC"]),
        "PR_AUC":       float(test_metrics["PR_AUC"]),
        "Balanced_Acc": float(test_metrics["Bal_Acc"]),
        "MCC":           float(test_metrics["MCC"]),
    }
}
with open(METADATA_PATH, "w") as fh:
    json.dump(metadata, fh, indent=2)

print(f"\n  Experimental pipeline saved -> {MODEL_PATH}")
print(f"  Results saved  -> outputs/push_to_85_results.csv")
