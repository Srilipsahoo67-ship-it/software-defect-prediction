"""Tune POI-specific defect classifiers without touching the held-out test set.

Model family, feature representation, and accuracy threshold are selected using
out-of-fold predictions from the training partition. The test partition is used
once, for the final performance report.
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from xgboost import XGBClassifier

from data_utils import normalize_bug_labels

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "datasets" / "poi-3.0.csv"
SEED = 42
FEATURES = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc", "ca", "ce",
    "npm", "lcom3", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc",
    "max_cc", "avg_cc",
]


def load_data(data_path=DATA_PATH):
    frame = pd.read_csv(data_path)
    frame.columns = [name.lower().strip() for name in frame.columns]
    label = next(name for name in frame.columns if name in {"bug", "defect", "class", "label"})
    frame = frame.rename(columns={label: "bug"})
    frame["bug"] = normalize_bug_labels(frame["bug"])
    frame = frame.dropna(subset=["bug"])
    frame["bug"] = frame["bug"].astype(int)
    features = [name for name in FEATURES if name in frame.columns]
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame[name] = frame[name].replace([np.inf, -np.inf], np.nan)
        frame[name] = frame[name].fillna(frame[name].median())
    # Rows are classes, not unique metric vectors. Distinct classes can share
    # the same measured metrics, so deduplicating feature/label pairs discards
    # valid training examples. The source file has no exact duplicate records.
    frame = frame[features + ["bug"]].reset_index(drop=True)
    return frame[features].to_numpy(dtype=float), frame["bug"].to_numpy(), features


def model_candidates():
    """Small, diverse parameter sweep suitable for this compact dataset."""
    return {
        "ExtraTrees_leaf1_sqrt": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "ExtraTrees_leaf2_sqrt": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "ExtraTrees_leaf2_all": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features=None, random_state=SEED, n_jobs=-1),
        "ExtraTrees_balanced": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "RandomForest_leaf1_sqrt": RandomForestClassifier(n_estimators=300, min_samples_leaf=1, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "RandomForest_leaf2_sqrt": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "RandomForest_leaf2_all": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features=None, random_state=SEED, n_jobs=-1),
        "RandomForest_balanced": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=SEED),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED)),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, l2_regularization=1.0, random_state=SEED),
        "LightGBM_leaf7": lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=7, min_child_samples=10, reg_lambda=2.0, verbosity=-1, random_state=SEED, n_jobs=-1),
        "LightGBM_leaf15": lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=10, reg_lambda=2.0, verbosity=-1, random_state=SEED, n_jobs=-1),
        "XGBoost_depth2": XGBClassifier(n_estimators=250, learning_rate=0.04, max_depth=2, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, eval_metric="logloss", random_state=SEED, n_jobs=1),
        "XGBoost_depth3": XGBClassifier(n_estimators=250, learning_rate=0.04, max_depth=3, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, eval_metric="logloss", random_state=SEED, n_jobs=1),
        "XGBoost_recall_weighted": XGBClassifier(n_estimators=250, learning_rate=0.04, max_depth=2, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, scale_pos_weight=0.8 / 0.2, reg_lambda=2.0, eval_metric="logloss", random_state=SEED, n_jobs=1),
        "LightGBM_recall_weighted": lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=7, min_child_samples=10, is_unbalance=True, reg_lambda=2.0, verbosity=-1, random_state=SEED, n_jobs=-1),
    }


def feature_views(X, features):
    """Compare CK-only, all raw metrics, and all raw plus row-wise features."""
    ck_names = {"wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"}
    ck = X[:, [index for index, name in enumerate(features) if name in ck_names]]
    index = {name: i for i, name in enumerate(features)}
    wmc, loc = X[:, index["wmc"]], X[:, index["loc"]]
    cbo, rfc, lcom = X[:, index["cbo"]], X[:, index["rfc"]], X[:, index["lcom"]]
    derived = np.column_stack([
        np.log1p(np.maximum(wmc, 0)), np.log1p(np.maximum(loc, 0)),
        np.log1p(np.maximum(lcom, 0)), np.log1p(np.maximum(rfc, 0)),
        np.log1p(np.maximum(cbo, 0)), wmc / (loc + 1e-6),
        cbo / (loc + 1e-6), rfc / (wmc + 1e-6),
        lcom / (wmc + 1e-6), cbo + rfc,
    ])
    return {
        "CK_7": ck,
        "All_raw_20": X,
        "All_raw_plus_engineered": np.column_stack([X, derived]),
    }


def select_threshold(probabilities, y_true, objective):
    thresholds = np.unique(np.r_[0.05, np.arange(0.10, 0.91, 0.01), 0.95, 0.5])
    if objective == "f1":
        scored = [(f1_score(y_true, probabilities >= threshold, zero_division=0), threshold)
                  for threshold in thresholds]
    else:
        scored = [(accuracy_score(y_true, probabilities >= threshold), threshold)
                  for threshold in thresholds]
    return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]


def main(objective="accuracy", split_seed=42):
    suffix = f"_seed{split_seed}" if split_seed != 42 else ""
    model_path = BASE / "models" / f"poi_{objective}_model{suffix}.pkl"
    result_path = BASE / "outputs" / f"poi_{objective}_results{suffix}.csv"
    X, y, features = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=split_seed,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    training_views = feature_views(X_train, features)
    test_views = feature_views(X_test, features)
    summaries = []

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    cv_majority_prediction = np.full(len(y_train), int(np.mean(y_train) >= 0.5))
    baseline_probability = baseline.predict_proba(X_test)[:, 1]
    baseline_prediction = baseline.predict(X_test)
    baseline_row = {
        "Model": "Majority_baseline",
        "Threshold": 0.5,
        "CV_Accuracy": max(np.mean(y_train), 1.0 - np.mean(y_train)),
        "CV_F1": f1_score(y_train, cv_majority_prediction, zero_division=0),
        "CV_Recall": recall_score(y_train, cv_majority_prediction, zero_division=0),
        "Test_Accuracy": accuracy_score(y_test, baseline_prediction),
        "Test_Balanced_Accuracy": balanced_accuracy_score(y_test, baseline_prediction),
        "Test_Precision": precision_score(y_test, baseline_prediction, zero_division=0),
        "Test_Recall": recall_score(y_test, baseline_prediction, zero_division=0),
        "Test_F1": f1_score(y_test, baseline_prediction, zero_division=0),
        "Test_ROC_AUC": roc_auc_score(y_test, baseline_probability),
    }
    summaries.append((baseline_row, baseline, "All_raw_20"))
    print("{Model:42s} cv_acc={CV_Accuracy:.3f} test_acc={Test_Accuracy:.3f}".format(**baseline_row))

    for view_name, X_view_train in training_views.items():
        for model_name in model_candidates():
            name = f"{model_name}__{view_name}"
            oof = cross_val_predict(
                model_candidates()[model_name], X_view_train, y_train,
                cv=cv, method="predict_proba", n_jobs=1,
            )[:, 1]
            threshold = select_threshold(oof, y_train, objective)
            cv_prediction = (oof >= threshold).astype(int)
            cv_accuracy = accuracy_score(y_train, cv_prediction)
            fitted = model_candidates()[model_name]
            fitted.fit(X_view_train, y_train)
            probability = fitted.predict_proba(test_views[view_name])[:, 1]
            prediction = (probability >= threshold).astype(int)
            row = {
                "Model": name,
                "Threshold": threshold,
                "CV_Accuracy": cv_accuracy,
                "CV_F1": f1_score(y_train, cv_prediction, zero_division=0),
                "CV_Recall": recall_score(y_train, cv_prediction, zero_division=0),
                "Test_Accuracy": accuracy_score(y_test, prediction),
                "Test_Recall": recall_score(y_test, prediction, zero_division=0),
                "Test_Balanced_Accuracy": balanced_accuracy_score(y_test, prediction),
                "Test_Precision": precision_score(y_test, prediction, zero_division=0),
                "Test_F1": f1_score(y_test, prediction, zero_division=0),
                "Test_ROC_AUC": roc_auc_score(y_test, probability),
            }
            summaries.append((row, fitted, view_name))
            print("{Model:42s} cv_acc={CV_Accuracy:.3f} test_acc={Test_Accuracy:.3f} "
                  "test_f1={Test_F1:.3f} recall={Test_Recall:.3f} "
                  "threshold={Threshold:.2f}".format(**row))

    selection_metric = "CV_F1" if objective == "f1" else "CV_Accuracy"
    best_row, best_model, best_view = max(summaries, key=lambda result: result[0][selection_metric])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item[0] for item in summaries]).sort_values(
        selection_metric, ascending=False,
    ).to_csv(result_path, index=False)
    joblib.dump({
        "model": best_model,
        "features": features,
        "feature_view": best_view,
        "threshold": best_row["Threshold"],
        "model_name": best_row["Model"],
        "objective": objective,
        "cv_accuracy": best_row["CV_Accuracy"],
        "cv_f1": best_row["CV_F1"],
        "test_metrics": {key: value for key, value in best_row.items() if key.startswith("Test_")},
        "seed": SEED,
        "split_seed": split_seed,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }, model_path)
    print(f"\nSelected on 5-fold training CV {objective}: {best_row['Model']}")
    print(f"Training CV accuracy: {best_row['CV_Accuracy']:.3f}")
    print(f"Training CV F1 / recall: {best_row['CV_F1']:.3f} / {best_row['CV_Recall']:.3f}")
    print(f"Held-out test accuracy: {best_row['Test_Accuracy']:.3f}")
    print(f"Held-out test F1 / recall: {best_row['Test_F1']:.3f} / {best_row['Test_Recall']:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved comparison: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=["accuracy", "f1"], default="accuracy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split and CV folds.")
    args = parser.parse_args()
    main(args.objective, split_seed=args.seed)
