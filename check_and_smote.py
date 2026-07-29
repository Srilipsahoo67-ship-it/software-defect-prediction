import os, glob, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# ── 1. Check jedit columns ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: jedit-4.3.csv column check")
print("=" * 60)
jedit = pd.read_csv(os.path.join(DATASET_DIR, "jedit-4.3.csv"))
jedit.columns = [c.lower().strip() for c in jedit.columns]
print(f"Shape: {jedit.shape}")
print(f"Columns: {list(jedit.columns)}")
ck_present = [c for c in CK_FEATURES if c in jedit.columns]
ck_missing = [c for c in CK_FEATURES if c not in jedit.columns]
print(f"\nCK features present : {ck_present}")
print(f"CK features missing : {ck_missing}")
label = next((c for c in jedit.columns if c in ["bug", "defect", "class", "label"]), None)
print(f"Defect label column : {label}")
print(f"Sample values of '{label}': {jedit[label].value_counts().to_dict()}")

# ── 2. Load all datasets ─────────────────────────────────────────────────────
def load_all():
    frames = []
    for f in glob.glob(os.path.join(DATASET_DIR, "*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        lbl = next((c for c in df.columns if c in ["bug", "defect", "class", "label"]), None)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)
print(f"\nBefore SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"After  SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM"          : SVC(probability=True, random_state=42),
        "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
    }

def evaluate_all(X_tr, y_tr, label):
    models = get_models()
    rows = []
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
    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")
    print(res.to_string())
    return res, models

# ── 3. Without SMOTE ─────────────────────────────────────────────────────────
res_no_smote, _ = evaluate_all(X_train_sc, y_train, "WITHOUT SMOTE")

# ── 4. With SMOTE ────────────────────────────────────────────────────────────
res_smote, trained_smote = evaluate_all(X_train_sm, y_train_sm, "WITH SMOTE")

# ── 5. Save comparison CSV ───────────────────────────────────────────────────
res_no_smote.to_csv(os.path.join(SCRIPT_DIR, "results_no_smote.csv"))
res_smote.to_csv(os.path.join(SCRIPT_DIR, "results_smote.csv"))
print("\nSaved: results_no_smote.csv and results_smote.csv")

# ── 6. Plot SMOTE comparison ─────────────────────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

res_no_smote[metrics].plot(kind="bar", ax=axes[0], rot=15)
axes[0].set_title("Without SMOTE")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="lower right")

res_smote[metrics].plot(kind="bar", ax=axes[1], rot=15)
axes[1].set_title("With SMOTE")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].legend(loc="lower right")

plt.suptitle("Model Comparison: Without vs With SMOTE", fontsize=14, fontweight="bold")
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "smote_comparison.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ── 7. Feature importance (RF with SMOTE) ────────────────────────────────────
rf_smote = trained_smote["Random Forest"]
imp = pd.Series(rf_smote.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
imp.plot(kind="barh", ax=ax, color="#5C85D6")
ax.set_title("Feature Importance - Random Forest (with SMOTE)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "feature_importance_smote.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

print("\nAll checks and SMOTE comparison complete.")
