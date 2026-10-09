"""Build the XGBoost model used by the 20-metric Streamlit demo.

This exports the recorded XGBoost_depth2__All_raw_20 candidate using the same
training split and threshold as outputs/poi_f1_results_seed2026.csv. It does
not rewrite the evaluation CSV or train on its held-out rows.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from train_poi_model import BASE, FEATURES, model_candidates, load_data


RESULTS_PATH = BASE / "outputs" / "poi_f1_results_seed2026.csv"
MODEL_PATH = BASE / "models" / "poi_xgboost_app.pkl"
MODEL_ROW = "XGBoost_depth2__All_raw_20"
SPLIT_SEED = 2026


def main():
    results = pd.read_csv(RESULTS_PATH)
    matching = results.loc[results["Model"] == MODEL_ROW]
    if len(matching) != 1:
        raise ValueError(f"Expected one {MODEL_ROW!r} row in {RESULTS_PATH}; found {len(matching)}")
    recorded = matching.iloc[0]

    X, y, features = load_data()
    if features != FEATURES:
        raise ValueError(f"Expected all 20 POI features; found {features}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED,
    )

    model = model_candidates()["XGBoost_depth2"]
    model.fit(X_train, y_train)
    test_probability = model.predict_proba(X_test)[:, 1]
    threshold = float(recorded["Threshold"])
    test_prediction = (test_probability >= threshold).astype(int)
    metrics = {
        "Test_Accuracy": accuracy_score(y_test, test_prediction),
        "Test_Precision": precision_score(y_test, test_prediction, zero_division=0),
        "Test_Recall": recall_score(y_test, test_prediction, zero_division=0),
        "Test_F1": f1_score(y_test, test_prediction, zero_division=0),
        "Test_ROC_AUC": roc_auc_score(y_test, test_probability),
    }
    for name, value in metrics.items():
        if not np.isclose(value, float(recorded[name]), atol=1e-12):
            raise RuntimeError(
                f"Recreated metric {name}={value:.12f} differs from the recorded "
                f"result {float(recorded[name]):.12f}; model not exported."
            )

    bundle = {
        "model": model,
        "model_name": "XGBoost",
        "candidate": MODEL_ROW,
        "features": features,
        "threshold": threshold,
        "selection_rule": "Highest XGBoost training CV accuracy; CV F1 breaks the tie",
        "split_seed": SPLIT_SEED,
        "training_rows": len(y_train),
        "test_rows": len(y_test),
        "cv_accuracy": float(recorded["CV_Accuracy"]),
        "cv_f1": float(recorded["CV_F1"]),
        "test_metrics": metrics,
        "input_defaults": dict(zip(features, np.median(X_train, axis=0).astype(float))),
        "sklearn_version": __import__("sklearn").__version__,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved panel demo model: {MODEL_PATH}")
    print(f"Candidate: {MODEL_ROW}; threshold: {threshold:.2f}")
    print("Held-out metrics:")
    for name, value in metrics.items():
        print(f"  {name.removeprefix('Test_')}: {value:.4f}")


if __name__ == "__main__":
    main()
