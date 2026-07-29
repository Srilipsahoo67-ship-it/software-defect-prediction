"""
Step 8 — Hyperparameter Tuning (GridSearchCV)
Models  : Random Forest, XGBoost
CV      : 5-Fold Stratified
Sampling: SMOTE inside each fold (no leakage)
Metric  : F1-score
"""

import os, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# ── Load & Preprocess ────────────────────────────────────────────────────────

def load_all():
    frames = []
    for f in glob.glob(os.path.join(DATASET_DIR, "*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
        if lbl:
            df = df.rename(columns={lbl: "bug"})
            frames.append(df)
    combined  = pd.concat(frames, ignore_index=True)
    available = [c for c in CK_FEATURES if c in combined.columns]
    combined  = combined[available + ["bug"]].copy()
    combined["bug"] = combined["bug"].astype(str).str.lower().str.strip()
    combined["bug"] = combined["bug"].map({"true":1,"false":0,"yes":1,"no":0,"1":1,"0":0})
    combined = combined.dropna(subset=["bug"])
    combined["bug"] = combined["bug"].astype(int)
    for col in available:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].median())
    return combined.drop_duplicates(), available

df, features = load_all()
X = df[features].values
y = df["bug"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer  = make_scorer(f1_score, zero_division=0)

# ── Helper ───────────────────────────────────────────────────────────────────

def print_results(name, grid, X_test, y_test):
    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"  {name} — Tuning Results")
    print(f"{'='*60}")
    print(f"  Best Parameters : {grid.best_params_}")
    print(f"  Best CV F1      : {grid.best_score_:.4f}")
    print(f"  Test Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test F1         : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Test AUC-ROC    : {roc_auc_score(y_test, y_prob):.4f}")
    return best

# ── Random Forest Tuning ─────────────────────────────────────────────────────

print("=" * 60)
print("STEP 8 — Hyperparameter Tuning with GridSearchCV + SMOTE")
print("=" * 60)

print("\n[1/2] Tuning Random Forest ...")

rf_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("model",  RandomForestClassifier(random_state=42)),
])

rf_grid = {
    "model__n_estimators"   : [100, 200, 300],
    "model__max_depth"      : [10, 20, None],
    "model__min_samples_split": [2, 5, 10],
}

rf_search = GridSearchCV(
    rf_pipe, rf_grid, cv=skf, scoring=scorer,
    n_jobs=-1, verbose=1, refit=True
)
rf_search.fit(X_train, y_train)
best_rf = print_results("Random Forest", rf_search, X_test, y_test)

# ── XGBoost Tuning ───────────────────────────────────────────────────────────

print("\n[2/2] Tuning XGBoost ...")

xgb_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("model",  XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False)),
])

xgb_grid = {
    "model__n_estimators" : [100, 200],
    "model__learning_rate": [0.01, 0.1],
    "model__max_depth"    : [3, 5, 7],
}

xgb_search = GridSearchCV(
    xgb_pipe, xgb_grid, cv=skf, scoring=scorer,
    n_jobs=-1, verbose=1, refit=True
)
xgb_search.fit(X_train, y_train)
best_xgb = print_results("XGBoost", xgb_search, X_test, y_test)

# ── Comparison Table ─────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  TUNED MODELS — FINAL COMPARISON")
print(f"{'='*60}")

rows = []
for name, search in [("Random Forest (Tuned)", rf_search), ("XGBoost (Tuned)", xgb_search)]:
    best   = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    rows.append({
        "Model"    : name,
        "Best CV F1" : round(search.best_score_, 4),
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC-ROC"  : round(roc_auc_score(y_test, y_prob), 4),
    })

cmp_df = pd.DataFrame(rows).set_index("Model")
print(cmp_df.to_string())
cmp_df.to_csv(os.path.join(SCRIPT_DIR, "results_tuned.csv"))
print(f"\nSaved: results_tuned.csv")

# ── Plot: Tuned vs Baseline ───────────────────────────────────────────────────

baseline = pd.read_csv(os.path.join(SCRIPT_DIR, "results_smote.csv"), index_col="Model")

metrics = ["F1", "AUC-ROC", "Accuracy"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric in zip(axes, metrics):
    vals = {
        "Baseline (SMOTE)": [
            baseline.loc["Random Forest", metric],
            baseline.loc["XGBoost", metric],
        ],
        "Tuned": [
            cmp_df.loc["Random Forest (Tuned)", metric],
            cmp_df.loc["XGBoost (Tuned)", metric],
        ],
    }
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, vals["Baseline (SMOTE)"], width, label="Baseline", color="#5C85D6")
    ax.bar(x + width/2, vals["Tuned"],            width, label="Tuned",    color="#F4845F")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random Forest", "XGBoost"], rotation=10)
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.legend()

plt.suptitle("Baseline vs Tuned Models (SMOTE + GridSearchCV)", fontsize=13, fontweight="bold")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "tuned_comparison.png")
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved: {p}")

print("\nStep 8 complete.")
