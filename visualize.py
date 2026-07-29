"""
Software Defect Prediction - Visualizations
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves without display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid", palette="muted")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# ── Load & Preprocess ────────────────────────────────────────────────────────

def load_and_prep():
    files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        label = next((c for c in df.columns if c in ["bug", "defect", "class", "label"]), None)
        if label:
            df = df.rename(columns={label: "bug"})
            frames.append(df)

    combined  = pd.concat(frames, ignore_index=True)
    available = [c for c in CK_FEATURES if c in combined.columns]
    combined  = combined[available + ["bug"]].copy()

    combined["bug"] = combined["bug"].astype(str).str.lower().str.strip()
    combined["bug"] = combined["bug"].map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    combined = combined.dropna(subset=["bug"])
    combined["bug"] = combined["bug"].astype(int)

    for col in available:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].median())

    return combined.drop_duplicates(), available


df, features = load_and_prep()
X = df[features].values
y = df["bug"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM"          : SVC(probability=True, random_state=42),
    "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
}
print("Training models for visualization...")
for name, m in models.items():
    m.fit(X_train, y_train)
    print(f"  {name} done")

# ── 1. Class Distribution ────────────────────────────────────────────────────

def plot_class_distribution():
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["No Defect (0)", "Defect (1)"], counts.values, color=["#4CAF50", "#F44336"], width=0.4)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── 2. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap():
    corr = df[features + ["bug"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── 3. Confusion Matrices ────────────────────────────────────────────────────

def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, model) in zip(axes.flatten(), models.items()):
        cm   = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Defect", "Defect"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── 4. ROC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, model), color in zip(models.items(), colors):
        y_prob      = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── 5. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key in zip(axes, ["Random Forest", "XGBoost"]):
        importances = models[key].feature_importances_
        idx = np.argsort(importances)[::-1]
        ax.bar(range(len(features)), importances[idx], color="#5C85D6", edgecolor="white")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([features[i].upper() for i in idx], rotation=30)
        ax.set_title(f"Feature Importance - {key}")
        ax.set_ylabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── 6. Model Comparison Bar Chart ────────────────────────────────────────────

def plot_model_comparison():
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        rows.append({
            "Model"   : name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall"  : recall_score(y_test, y_pred, zero_division=0),
            "F1"      : f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC" : roc_auc_score(y_test, y_prob),
        })
    res = pd.DataFrame(rows).set_index("Model")
    ax  = res[metrics].plot(kind="bar", figsize=(12, 5), rot=15)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── Run All ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating plots...\n")
    plot_class_distribution()
    plot_correlation_heatmap()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_feature_importance()
    plot_model_comparison()
    print(f"\nAll plots saved in: {PLOTS_DIR}")
