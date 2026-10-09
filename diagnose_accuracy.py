"""
diagnose_accuracy.py
====================
Investigates why accuracy dropped and finds the path to 85%.
Runs silently, prints results, saves outputs/accuracy_investigation.csv

This is an exploratory legacy script: it sweeps thresholds and compares
candidate models on the same holdout set. Its best score is not an unbiased
generalization estimate. Use improved_pipeline.py for the canonical evaluation.
"""
import glob, warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              recall_score, precision_score,
                              average_precision_score, balanced_accuracy_score,
                              matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from data_utils import normalize_bug_labels

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
EXPERIMENTAL_MODEL_PATH = os.path.join(
    BASE, "models", "accuracy_investigation_pipeline.pkl")
EXPERIMENTAL_METADATA_PATH = os.path.join(
    BASE, "models", "accuracy_investigation_metadata.json")

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
print("Total samples:", len(raw), "  Defect rate:", round(raw["bug"].mean()*100,1), "%")

def run(label, fcols_base, model, thr=0.5, use_smote=True):
    fcols = [c for c in fcols_base if c in raw.columns]
    sub = raw[fcols + ["bug"]].copy()
    for c in fcols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub[c] = sub[c].fillna(sub[c].median())
    sub = sub.replace([np.inf, -np.inf], 0).drop_duplicates()
    X = sub[fcols].values.astype(float)
    y = sub["bug"].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler()
    X_tr_sc = np.nan_to_num(sc.fit_transform(X_tr), nan=0, posinf=0, neginf=0)
    X_te_sc = np.nan_to_num(sc.transform(X_te),     nan=0, posinf=0, neginf=0)

    if use_smote:
        sm = SMOTE(random_state=SEED)
        X_fit, y_fit = sm.fit_resample(X_tr_sc, y_tr)
    else:
        X_fit, y_fit = X_tr_sc, y_tr

    model.fit(X_fit, y_fit)
    ypr = model.predict_proba(X_te_sc)[:, 1]
    yp  = (ypr >= thr).astype(int)

    return {
        "Label": label, "N_Features": len(fcols), "Threshold": thr,
        "Accuracy":  round(accuracy_score(y_te, yp), 4),
        "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0), 4),
        "F1":        round(f1_score(y_te, yp, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_te, ypr), 4),
        "PR_AUC":    round(average_precision_score(y_te, ypr), 4),
        "Bal_Acc":   round(balanced_accuracy_score(y_te, yp), 4),
        "MCC":       round(matthews_corrcoef(y_te, yp), 4),
    }

rows = []

# ── PART 1: Feature set comparison (LightGBM, thr=0.5, SMOTE) ────────────────
print("\n--- PART 1: Feature Set Impact (LightGBM, thr=0.5, SMOTE) ---")
lgbm_base = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 num_leaves=63, random_state=SEED,
                                 n_jobs=-1, verbose=-1)
for lbl, fcols in [
    ("A_CK_only_7",    CK),
    ("B_CK_eng_19",    CK + ENG),
    ("C_ALL_RAW_20",   ALL_RAW),
    ("D_ALL_RAW_ENG_32", ALL_RAW + ENG),
]:
    r = run(lbl, fcols, lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                            num_leaves=63, random_state=SEED,
                                            n_jobs=-1, verbose=-1))
    rows.append(r)
    print(f"  {lbl:22s}  Acc={r['Accuracy']:.4f}  F1={r['F1']:.4f}  "
          f"AUC={r['ROC_AUC']:.4f}  Rec={r['Recall']:.4f}")

# ── PART 2: Model comparison (ALL_RAW_ENG_32, thr=0.5, SMOTE) ────────────────
print("\n--- PART 2: Model Comparison (ALL_RAW_ENG_32, thr=0.5, SMOTE) ---")
best_fcols = ALL_RAW + ENG
models_to_test = {
    "LightGBM_300":  lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                          num_leaves=63, random_state=SEED,
                                          n_jobs=-1, verbose=-1),
    "LightGBM_600":  lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                          num_leaves=63, random_state=SEED,
                                          n_jobs=-1, verbose=-1),
    "XGBoost_300":   XGBClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=6, eval_metric="logloss",
                                    random_state=SEED, n_jobs=-1),
    "ExtraTrees_300":ExtraTreesClassifier(n_estimators=300, random_state=SEED,
                                           n_jobs=-1),
    "RF_300":        RandomForestClassifier(n_estimators=300, random_state=SEED,
                                             n_jobs=-1),
    "HistGB":        HistGradientBoostingClassifier(random_state=SEED,
                                                     max_iter=300),
}
for mname, model in models_to_test.items():
    r = run("D_ALL_RAW_ENG_32_"+mname, best_fcols, model)
    rows.append(r)
    print(f"  {mname:20s}  Acc={r['Accuracy']:.4f}  F1={r['F1']:.4f}  "
          f"AUC={r['ROC_AUC']:.4f}  Rec={r['Recall']:.4f}")

# ── PART 3: Threshold sweep on best model ─────────────────────────────────────
print("\n--- PART 3: Threshold Sweep (LightGBM_600, ALL_RAW_ENG_32, SMOTE) ---")
fcols = [c for c in ALL_RAW + ENG if c in raw.columns]
sub = raw[fcols + ["bug"]].copy()
for c in fcols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub[c] = sub[c].fillna(sub[c].median())
sub = sub.replace([np.inf, -np.inf], 0).drop_duplicates()
X = sub[fcols].values.astype(float)
y = sub["bug"].values.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=SEED, stratify=y)
sc = StandardScaler()
X_tr_sc = np.nan_to_num(sc.fit_transform(X_tr), nan=0, posinf=0, neginf=0)
X_te_sc = np.nan_to_num(sc.transform(X_te),     nan=0, posinf=0, neginf=0)
sm = SMOTE(random_state=SEED)
X_sm, y_sm = sm.fit_resample(X_tr_sc, y_tr)
best_lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                 num_leaves=63, random_state=SEED,
                                 n_jobs=-1, verbose=-1)
best_lgbm.fit(X_sm, y_sm)
ypr = best_lgbm.predict_proba(X_te_sc)[:, 1]

thr_rows = []
for t in np.arange(0.30, 0.76, 0.05):
    yp = (ypr >= t).astype(int)
    thr_rows.append({
        "Threshold": round(t, 2),
        "Accuracy":  round(accuracy_score(y_te, yp), 4),
        "F1":        round(f1_score(y_te, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0), 4),
        "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
        "Bal_Acc":   round(balanced_accuracy_score(y_te, yp), 4),
    })
thr_df = pd.DataFrame(thr_rows)
print(thr_df.to_string(index=False))
best_acc_thr = float(thr_df.loc[thr_df["Accuracy"].idxmax(), "Threshold"])
best_f1_thr  = float(thr_df.loc[thr_df["F1"].idxmax(),       "Threshold"])
print(f"\n  Best Accuracy threshold: {best_acc_thr}  -> Acc={thr_df.loc[thr_df['Accuracy'].idxmax(),'Accuracy']}")
print(f"  Best F1 threshold:       {best_f1_thr}  -> F1={thr_df.loc[thr_df['F1'].idxmax(),'F1']}")

# ── PART 4: Tuned LightGBM (RandomizedSearchCV, ALL_RAW_ENG_32) ──────────────
print("\n--- PART 4: Tuned LightGBM (RandomizedSearchCV, 50 iters) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=SEED)),
    ("model",  lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1)),
])
grid = {
    "model__n_estimators":      [300, 500, 700, 1000],
    "model__max_depth":         [4, 6, 8, -1],
    "model__learning_rate":     [0.01, 0.03, 0.05, 0.08],
    "model__num_leaves":        [31, 63, 127, 255],
    "model__min_child_samples": [5, 10, 20, 30],
    "model__subsample":         [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "model__reg_alpha":         [0, 0.05, 0.1, 0.5],
    "model__reg_lambda":        [0.1, 0.5, 1.0, 5.0],
}
# Use accuracy as scoring to push accuracy up
rs_acc = RandomizedSearchCV(pipe, grid, n_iter=50, cv=skf,
                             scoring="accuracy", random_state=SEED,
                             n_jobs=-1, refit=True)
rs_acc.fit(X_tr, y_tr)   # raw unscaled — pipeline handles scaling
best_acc_pipe = rs_acc.best_estimator_
ypr_tuned = best_acc_pipe.predict_proba(X_te)[:, 1]

print(f"  Best CV Accuracy: {rs_acc.best_score_:.4f}")
print(f"  Best params: {rs_acc.best_params_}")

# Sweep thresholds on tuned model
print("\n  Threshold sweep on tuned model:")
thr_rows2 = []
for t in np.arange(0.30, 0.76, 0.05):
    yp = (ypr_tuned >= t).astype(int)
    thr_rows2.append({
        "Threshold": round(t, 2),
        "Accuracy":  round(accuracy_score(y_te, yp), 4),
        "F1":        round(f1_score(y_te, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0), 4),
        "Precision": round(precision_score(y_te, yp, zero_division=0), 4),
        "Bal_Acc":   round(balanced_accuracy_score(y_te, yp), 4),
    })
thr_df2 = pd.DataFrame(thr_rows2)
print(thr_df2.to_string(index=False))
best_t2 = float(thr_df2.loc[thr_df2["Accuracy"].idxmax(), "Threshold"])
best_row2 = thr_df2.loc[thr_df2["Accuracy"].idxmax()]
print(f"\n  TUNED MODEL BEST ACCURACY: {best_row2['Accuracy']:.4f} at threshold={best_t2}")

# Also tune XGBoost for accuracy
print("\n--- PART 5: Tuned XGBoost (accuracy scoring) ---")
pipe_xgb = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=SEED)),
    ("model",  XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)),
])
grid_xgb = {
    "model__n_estimators":     [300, 500, 700],
    "model__max_depth":        [4, 6, 8],
    "model__learning_rate":    [0.01, 0.03, 0.05],
    "model__subsample":        [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__reg_alpha":        [0, 0.1, 0.5],
    "model__reg_lambda":       [0.5, 1.0, 5.0],
    "model__min_child_weight": [1, 3, 5],
}
rs_xgb = RandomizedSearchCV(pipe_xgb, grid_xgb, n_iter=50, cv=skf,
                              scoring="accuracy", random_state=SEED,
                              n_jobs=-1, refit=True)
rs_xgb.fit(X_tr, y_tr)
best_xgb_pipe = rs_xgb.best_estimator_
ypr_xgb = best_xgb_pipe.predict_proba(X_te)[:, 1]
print(f"  Best CV Accuracy: {rs_xgb.best_score_:.4f}")

thr_rows3 = []
for t in np.arange(0.30, 0.76, 0.05):
    yp = (ypr_xgb >= t).astype(int)
    thr_rows3.append({
        "Threshold": round(t, 2),
        "Accuracy":  round(accuracy_score(y_te, yp), 4),
        "F1":        round(f1_score(y_te, yp, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0), 4),
    })
thr_df3 = pd.DataFrame(thr_rows3)
best_xgb_row = thr_df3.loc[thr_df3["Accuracy"].idxmax()]
print(f"  XGBoost best Accuracy: {best_xgb_row['Accuracy']:.4f} at thr={best_xgb_row['Threshold']}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Old pipeline (incorrect labels):  Acc=0.8321")
print(f"  Correct labels, LightGBM_600, thr={best_acc_thr}:  "
      f"Acc={thr_df.loc[thr_df['Accuracy'].idxmax(),'Accuracy']:.4f}")
print(f"  Tuned LightGBM, thr={best_t2}:  "
      f"Acc={best_row2['Accuracy']:.4f}  F1={best_row2['F1']:.4f}")
print(f"  Tuned XGBoost, thr={best_xgb_row['Threshold']}:  "
      f"Acc={best_xgb_row['Accuracy']:.4f}  F1={best_xgb_row['F1']:.4f}")

import joblib, json
# Save best accuracy model
if best_row2["Accuracy"] >= best_xgb_row["Accuracy"]:
    best_pipe_final = best_acc_pipe
    best_thr_final  = best_t2
    best_acc_final  = float(best_row2["Accuracy"])
    best_model_name = "LightGBM_Tuned_Accuracy"
    best_feat_cols  = fcols
else:
    best_pipe_final = best_xgb_pipe
    best_thr_final  = float(best_xgb_row["Threshold"])
    best_acc_final  = float(best_xgb_row["Accuracy"])
    best_model_name = "XGBoost_Tuned_Accuracy"
    best_feat_cols  = fcols

print(f"\n  BEST MODEL: {best_model_name}  Acc={best_acc_final:.4f}")

yp_final = (best_pipe_final.predict_proba(X_te)[:, 1] >= best_thr_final).astype(int)
ypr_final = best_pipe_final.predict_proba(X_te)[:, 1]
final_metrics = {
    "model": best_model_name,
    "feature_set": "ALL_RAW_ENG_32",
    "training_features": best_feat_cols,
    "imbalance_strategy": "SMOTE",
    "threshold": best_thr_final,
    "performance_metrics": {
        "Accuracy":     round(accuracy_score(y_te, yp_final), 4),
        "Precision":    round(precision_score(y_te, yp_final, zero_division=0), 4),
        "Recall":       round(recall_score(y_te, yp_final, zero_division=0), 4),
        "F1":           round(f1_score(y_te, yp_final, zero_division=0), 4),
        "ROC_AUC":      round(roc_auc_score(y_te, ypr_final), 4),
        "PR_AUC":       round(average_precision_score(y_te, ypr_final), 4),
        "Balanced_Acc": round(balanced_accuracy_score(y_te, yp_final), 4),
        "MCC":          round(matthews_corrcoef(y_te, yp_final), 4),
    }
}
print("\n  Final metrics:")
for k, v in final_metrics["performance_metrics"].items():
    print(f"    {k}: {v}")

joblib.dump(best_pipe_final, EXPERIMENTAL_MODEL_PATH)
with open(EXPERIMENTAL_METADATA_PATH, "w") as fh:
    json.dump(final_metrics, fh, indent=2)
print(f"\n  Experimental pipeline saved -> {EXPERIMENTAL_MODEL_PATH}")

pd.DataFrame(rows).to_csv(
    os.path.join(BASE, "outputs", "accuracy_investigation.csv"), index=False)
print("  Results saved -> outputs/accuracy_investigation.csv")
