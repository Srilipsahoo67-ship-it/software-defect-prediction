"""
Full Validation Pipeline:
1. No data leakage - SMOTE only on training set
2. Class distribution check
3. Confusion matrices (before/after SMOTE)
4. ROC curves (before/after SMOTE)
5. Feature importance
6. AUC comparison
7. 5-fold cross-validation
"""

import os, glob, warnings
warnings.filterwarnings("ignore")
from data_utils import normalize_bug_labels

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_theme(style="whitegrid")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# ── Load & Preprocess ────────────────────────────────────────────────────────

# Datasets excluded from training (extreme class imbalance — 98.4% defective)
EXCLUDE = ["xalan"]

def load_all():
    frames = []
    for f in glob.glob(os.path.join(DATASET_DIR, "*.csv")):
        if any(ex in os.path.basename(f).lower() for ex in EXCLUDE):
            continue
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        lbl = next((c for c in df.columns if c in ["bug","defect","class","label"]), None)
        if lbl:
            df = df.rename(columns={lbl: "bug"})
            frames.append(df)
    combined  = pd.concat(frames, ignore_index=True)
    available = [c for c in CK_FEATURES if c in combined.columns]
    combined  = combined[available + ["bug"]].copy()
    combined["bug"] = normalize_bug_labels(combined["bug"])
    combined = combined.dropna(subset=["bug"])
    combined["bug"] = combined["bug"].astype(int)
    for col in available:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].median())
    return combined.drop_duplicates(), available

df, features = load_all()
X = df[features].values
y = df["bug"].values

# ── STEP 1 & 2: Correct split → SMOTE only on train ─────────────────────────
print("=" * 60)
print("STEP 1 & 2: Train/Test Split + SMOTE on training set ONLY")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only
X_test_sc  = scaler.transform(X_test)        # transform test with train stats

print(f"\nTrain size : {X_train_sc.shape[0]} samples")
print(f"Test  size : {X_test_sc.shape[0]}  samples  <-- NEVER touched by SMOTE")

u, c = np.unique(y_train, return_counts=True)
print(f"\nClass distribution BEFORE SMOTE (train only):")
print(f"  No Bug (0): {c[0]}")
print(f"  Bug    (1): {c[1]}")

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)

u2, c2 = np.unique(y_train_sm, return_counts=True)
print(f"\nClass distribution AFTER SMOTE (train only):")
print(f"  No Bug (0): {c2[0]}")
print(f"  Bug    (1): {c2[1]}")
print(f"\nTest set class distribution (UNCHANGED):")
u3, c3 = np.unique(y_test, return_counts=True)
print(f"  No Bug (0): {c3[0]}")
print(f"  Bug    (1): {c3[1]}")
print("\nNo data leakage confirmed.")

# ── Model definitions ────────────────────────────────────────────────────────

def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM"          : SVC(probability=True, random_state=42),
        "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
    }

def train_eval(X_tr, y_tr, label):
    models = get_models()
    rows   = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp    = m.predict(X_test_sc)
        yprob = m.predict_proba(X_test_sc)[:, 1]
        rows.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, yp), 4),
            "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, yp, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, yp, zero_division=0), 4),
            "AUC-ROC"  : round(roc_auc_score(y_test, yprob), 4),
        })
    res = pd.DataFrame(rows).set_index("Model")
    print(f"\n{'='*60}\nRESULTS — {label}\n{'='*60}")
    print(res.to_string())
    return res, models

res_base,  models_base  = train_eval(X_train_sc, y_train,    "WITHOUT SMOTE")
res_smote, models_smote = train_eval(X_train_sm, y_train_sm, "WITH SMOTE")

# ── STEP 3: Confusion Matrices before & after SMOTE ─────────────────────────
print("\nSTEP 3: Generating confusion matrices...")

fig, axes = plt.subplots(4, 2, figsize=(14, 22))
model_names = list(models_base.keys())
for i, name in enumerate(model_names):
    for j, (models_dict, tag) in enumerate([(models_base, "No SMOTE"), (models_smote, "SMOTE")]):
        cm   = confusion_matrix(y_test, models_dict[name].predict(X_test_sc))
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Bug", "Bug"])
        disp.plot(ax=axes[i][j], colorbar=False, cmap="Blues")
        axes[i][j].set_title(f"{name} — {tag}")
        fn = cm[1][0]
        axes[i][j].set_xlabel(f"Predicted\n(False Negatives = {fn})")

plt.suptitle("Confusion Matrices: Without vs With SMOTE", fontsize=14, fontweight="bold")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "confusion_matrices_comparison.png")
plt.savefig(p, dpi=150); plt.close()
print(f"Saved: {p}")

# ── STEP 4: ROC Curves before & after SMOTE ─────────────────────────────────
print("STEP 4: Generating ROC curves...")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (models_dict, tag) in zip(axes, [(models_base, "Without SMOTE"), (models_smote, "With SMOTE")]):
    for (name, m), color in zip(models_dict.items(), colors):
        yprob       = m.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set(title=tag, xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)

plt.suptitle("ROC Curves: Without vs With SMOTE", fontsize=14, fontweight="bold")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "roc_curves_comparison.png")
plt.savefig(p, dpi=150); plt.close()
print(f"Saved: {p}")

# ── STEP 5: Feature Importance ───────────────────────────────────────────────
print("STEP 5: Generating feature importance...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
tree_models = ["Random Forest", "Decision Tree", "XGBoost"]

for ax, (name, m) in zip(axes.flatten(), models_smote.items()):
    if hasattr(m, "feature_importances_"):
        imp = pd.Series(m.feature_importances_, index=[f.upper() for f in features])
        imp = imp.sort_values(ascending=True)
        imp.plot(kind="barh", ax=ax, color="#5C85D6", edgecolor="white")
        ax.set_title(f"Feature Importance — {name} (SMOTE)")
        ax.set_xlabel("Importance Score")
    else:
        # SVM — use permutation importance approximation via coef not available,
        # show placeholder text
        ax.text(0.5, 0.5, "SVM: Feature importance\nnot directly available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(f"Feature Importance — {name}")

plt.tight_layout()
p = os.path.join(PLOTS_DIR, "feature_importance_all.png")
plt.savefig(p, dpi=150); plt.close()
print(f"Saved: {p}")

# ── STEP 6: AUC Comparison table ─────────────────────────────────────────────
print("\nSTEP 6: AUC-ROC Comparison")
print("=" * 60)
auc_compare = pd.DataFrame({
    "Without SMOTE": res_base["AUC-ROC"],
    "With SMOTE"   : res_smote["AUC-ROC"],
})
auc_compare["AUC Change"] = (auc_compare["With SMOTE"] - auc_compare["Without SMOTE"]).round(4)
print(auc_compare.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
auc_compare[["Without SMOTE", "With SMOTE"]].plot(kind="bar", ax=ax, rot=15, color=["#5C85D6","#F4845F"])
ax.set_title("AUC-ROC Comparison: Without vs With SMOTE")
ax.set_ylabel("AUC-ROC Score")
ax.set_ylim(0.4, 0.8)
ax.legend()
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "auc_comparison.png")
plt.savefig(p, dpi=150); plt.close()
print(f"Saved: {p}")

# ── STEP 7: 5-Fold Stratified Cross-Validation ───────────────────────────────
print("\nSTEP 7: 5-Fold Stratified Cross-Validation (with SMOTE in pipeline)")
print("=" * 60)

skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = []
for name, base_model in get_models().items():
    # ImbPipeline ensures SMOTE is applied inside each fold — no leakage
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=42)),
        ("model",  base_model),
    ])
    scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    cv_results.append({
        "Model"    : name,
        "Accuracy" : round(scores["test_accuracy"].mean(), 4),
        "Precision": round(scores["test_precision"].mean(), 4),
        "Recall"   : round(scores["test_recall"].mean(), 4),
        "F1"       : round(scores["test_f1"].mean(), 4),
        "AUC-ROC"  : round(scores["test_roc_auc"].mean(), 4),
    })
    print(f"  {name} done")

cv_df = pd.DataFrame(cv_results).set_index("Model")
print(f"\n{'='*60}")
print("5-FOLD CV RESULTS (mean scores, SMOTE inside each fold)")
print(f"{'='*60}")
print(cv_df.to_string())

cv_df.to_csv(os.path.join(OUTPUT_DIR, "results_cv.csv"))
print(f"\nSaved: outputs/results_cv.csv")

# Plot CV results
ax = cv_df.plot(kind="bar", figsize=(12, 6), rot=15)
ax.set_title("5-Fold Cross-Validation Results (with SMOTE)")
ax.set_ylabel("Mean Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "cv_results.png")
plt.savefig(p, dpi=150); plt.close()
print(f"Saved: {p}")

# ── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("\nRecall improvement after SMOTE:")
recall_compare = pd.DataFrame({
    "Without SMOTE": res_base["Recall"],
    "With SMOTE"   : res_smote["Recall"],
})
recall_compare["Improvement"] = (recall_compare["With SMOTE"] - recall_compare["Without SMOTE"]).round(4)
print(recall_compare.to_string())

print("\nBest model by F1 (with SMOTE):", res_smote["F1"].idxmax())
print("Best model by AUC (with SMOTE):", res_smote["AUC-ROC"].idxmax())
print("Best model by CV F1           :", cv_df["F1"].idxmax())
print("Best model by CV AUC          :", cv_df["AUC-ROC"].idxmax())
print("\nAll steps complete.")
