"""
visualize.py
============
Software Defect Prediction - Complete Visualization Suite

Generates:
  1.  class_distribution.png       - Class balance bar chart
  2.  correlation_heatmap.png      - Pearson correlation heatmap
  3.  confusion_matrices.png       - Confusion matrices for all models
  4.  roc_curves.png               - ROC curves for all models
  5.  precision_recall_curves.png  - Precision-Recall curves (better for imbalanced data)
  6.  feature_importance.png       - Feature importance (RF + XGBoost)
  7.  model_comparison.png         - Side-by-side metric comparison bar chart
  8.  dataset_statistics.png       - Dataset statistics summary panel
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid", palette="muted")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets")
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]
COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ── Load & Preprocess ────────────────────────────────────────────────────────

def load_and_prep():
    frames = []
    for f in glob.glob(os.path.join(DATASET_DIR, "*.csv")):
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
    combined["bug"] = combined["bug"].map({"true":1,"false":0,"yes":1,"no":0,"1":1,"0":0})
    combined = combined.dropna(subset=["bug"])
    combined["bug"] = combined["bug"].astype(int)
    for col in available:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].median())
    return combined.drop_duplicates(), available


df, features = load_and_prep()
X = df[features].values
y = df["bug"].values

# Train/test split → scale → SMOTE (consistent with train_models.py)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
sm         = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM"          : SVC(probability=True, random_state=42),
    "XGBoost"      : XGBClassifier(eval_metric="logloss", random_state=42),
}
print("Training models for visualization...")
for name, m in models.items():
    m.fit(X_train_sm, y_train_sm)
    print("  %s done" % name)

# ── 1. Dataset Statistics ────────────────────────────────────────────────────

def plot_dataset_statistics():
    total     = len(df)
    defective = int(df["bug"].sum())
    clean     = total - defective
    dup_raw   = sum(pd.read_csv(f).shape[0]
                    for f in glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    removed   = dup_raw - total

    stats = {
        "Total Samples"    : total,
        "Defective (Bug=1)": defective,
        "Clean (Bug=0)"    : clean,
        "Features Used"    : len(features),
        "Duplicates Removed": removed,
        "Missing Imputed"  : 0,
        "Train Samples"    : len(X_train_sm),
        "Test Samples"     : len(X_test_sc),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Statistics — PROMISE Repository", fontsize=14, fontweight="bold")

    # Left: stats table
    ax = axes[0]
    ax.axis("off")
    rows = [[k, str(v)] for k, v in stats.items()]
    tbl  = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                    cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        cell.set_edgecolor("#e2e8f0")

    # Right: class distribution pie
    ax2 = axes[1]
    ax2.pie([clean, defective],
            labels=["Clean\n%d (%.1f%%)" % (clean, 100*clean/total),
                    "Defective\n%d (%.1f%%)" % (defective, 100*defective/total)],
            colors=["#38a169", "#e53e3e"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11})
    ax2.set_title("Class Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "dataset_statistics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 2. Class Distribution Bar ────────────────────────────────────────────────

def plot_class_distribution():
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Clean (0)", "Defective (1)"], counts.values,
                  color=["#38a169", "#e53e3e"], width=0.4, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 3. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap():
    corr = df[features + ["bug"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 9}, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 4. Confusion Matrices ────────────────────────────────────────────────────

def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, model) in zip(axes.flatten(), models.items()):
        cm   = confusion_matrix(y_test, model.predict(X_test_sc))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Clean", "Defective"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        tn, fp, fn, tp = cm.ravel()
        ax.set_title("%s\nTP=%d  FP=%d  FN=%d  TN=%d" % (name, tp, fp, fn, tn),
                     fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices (with SMOTE)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 5. ROC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(models.items(), COLORS):
        y_prob      = model.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label="%s  (AUC = %.3f)" % (name, roc_auc))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — All Models (with SMOTE)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 6. Precision-Recall Curves ───────────────────────────────────────────────

def plot_precision_recall_curves():
    """
    Precision-Recall curves are more informative than ROC for imbalanced datasets.
    A high area under PR curve means both precision and recall are high.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.sum() / len(y_test)   # random classifier baseline

    for (name, model), color in zip(models.items(), COLORS):
        y_prob        = model.predict_proba(X_test_sc)[:, 1]
        prec, rec, _  = precision_recall_curve(y_test, y_prob)
        ap            = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=color, lw=2.5,
                label="%s  (AP = %.3f)" % (name, ap))

    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1.5,
               label="Random Classifier (AP = %.2f)" % baseline)
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves — All Models (with SMOTE)",
           xlim=[0, 1], ylim=[0, 1.05])
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 7. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance():
    """
    Horizontal bar chart showing feature importance from RF and XGBoost.
    Sorted descending so most important feature is at the top.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    feat_labels = [f.upper() for f in features]

    for ax, key in zip(axes, ["Random Forest", "XGBoost"]):
        importances = models[key].feature_importances_
        sorted_idx  = np.argsort(importances)          # ascending for barh
        sorted_imp  = importances[sorted_idx]
        sorted_lbl  = [feat_labels[i] for i in sorted_idx]

        bars = ax.barh(sorted_lbl, sorted_imp,
                       color="#5C85D6", edgecolor="white", height=0.6)
        # Value labels
        for bar, val in zip(bars, sorted_imp):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    "%.3f" % val, va="center", ha="left", fontsize=9, color="#2d3748")

        ax.set_title("Feature Importance — %s" % key, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(importances) * 1.25)

    plt.suptitle("CK Metric Feature Importance (with SMOTE)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── 8. Model Comparison Bar Chart ────────────────────────────────────────────

def plot_model_comparison():
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        rows.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "AUC-ROC"  : round(roc_auc_score(y_test, y_prob), 4),
        })

    res = pd.DataFrame(rows).set_index("Model")

    # Grouped bar chart
    x      = np.arange(len(res))
    width  = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    metric_colors = ["#4299e1", "#48bb78", "#ed8936", "#9f7aea", "#f56565"]

    for i, (metric, color) in enumerate(zip(metric_names, metric_colors)):
        offset = (i - 2) * width
        bars   = ax.bar(x + offset, res[metric], width, label=metric,
                        color=color, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(res.index, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — All Metrics (with SMOTE)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate best AUC
    best_model = res["AUC-ROC"].idxmax()
    best_val   = res["AUC-ROC"].max()
    ax.annotate("Best AUC: %s (%.4f)" % (best_model, best_val),
                xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ebf8ff", edgecolor="#3182ce"))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: %s" % path)

# ── Run All ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating all plots...\n")
    plot_dataset_statistics()
    plot_class_distribution()
    plot_correlation_heatmap()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_precision_recall_curves()
    plot_feature_importance()
    plot_model_comparison()
    print("\nAll plots saved in: %s" % PLOTS_DIR)
