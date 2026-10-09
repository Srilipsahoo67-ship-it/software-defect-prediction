"""Temporal POI validation: train on release 2.5, evaluate on release 3.0.

Model, feature view, and F1 threshold are selected using release 2.5 only.
Release 3.0 labels are used only for the final temporal evaluation.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train_poi_model import BASE, FEATURES, SEED, feature_views, load_data, model_candidates, select_threshold

TRAIN_PATH = BASE / "datasets" / "poi-2.5.csv"
TEST_PATH = BASE / "datasets" / "poi-3.0.csv"
RESULT_PATH = BASE / "outputs" / "poi_temporal_2.5_to_3.0_results.csv"
MODEL_PATH = BASE / "models" / "poi_temporal_model.pkl"


def main():
    X_train, y_train, train_features = load_data(TRAIN_PATH)
    X_test, y_test, test_features = load_data(TEST_PATH)
    if train_features != test_features:
        raise ValueError("POI releases do not have matching feature columns/order.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_views = feature_views(X_train, train_features)
    test_views = feature_views(X_test, test_features)
    results = []

    for view_name, X_fit in train_views.items():
        for model_name in model_candidates():
            oof_probability = cross_val_predict(
                model_candidates()[model_name], X_fit, y_train,
                cv=cv, method="predict_proba", n_jobs=1,
            )[:, 1]
            threshold = select_threshold(oof_probability, y_train, "f1")
            oof_prediction = (oof_probability >= threshold).astype(int)
            model = model_candidates()[model_name].fit(X_fit, y_train)

            probability = model.predict_proba(test_views[view_name])[:, 1]
            prediction = (probability >= threshold).astype(int)
            row = {
                "Model": f"{model_name}__{view_name}",
                "Threshold": threshold,
                "Train_CV_F1": f1_score(y_train, oof_prediction, zero_division=0),
                "Train_CV_Recall": recall_score(y_train, oof_prediction, zero_division=0),
                "Test_Accuracy": accuracy_score(y_test, prediction),
                "Test_Precision": precision_score(y_test, prediction, zero_division=0),
                "Test_Recall": recall_score(y_test, prediction, zero_division=0),
                "Test_F1": f1_score(y_test, prediction, zero_division=0),
                "Test_ROC_AUC": roc_auc_score(y_test, probability),
                "Test_Balanced_Accuracy": balanced_accuracy_score(y_test, prediction),
            }
            results.append((row, model, view_name))
            print("{Model:42s} CV_F1={Train_CV_F1:.3f} "
                  "3.0 accuracy={Test_Accuracy:.3f} recall={Test_Recall:.3f} "
                  "F1={Test_F1:.3f} AUC={Test_ROC_AUC:.3f}".format(**row))

    best_row, best_model, best_view = max(results, key=lambda item: item[0]["Train_CV_F1"])
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row for row, _, _ in results]).sort_values(
        "Train_CV_F1", ascending=False,
    ).to_csv(RESULT_PATH, index=False)
    joblib.dump({
        "model": best_model,
        "features": FEATURES,
        "feature_view": best_view,
        "threshold": best_row["Threshold"],
        "model_name": best_row["Model"],
        "training_release": "poi-2.5",
        "evaluation_release": "poi-3.0",
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "train_cv_f1": best_row["Train_CV_F1"],
        "test_metrics": {key: value for key, value in best_row.items() if key.startswith("Test_")},
    }, MODEL_PATH)

    print("\nSelected using POI 2.5 CV F1:", best_row["Model"])
    print("POI 3.0 temporal test — accuracy={Test_Accuracy:.3f}, precision={Test_Precision:.3f}, "
          "recall={Test_Recall:.3f}, F1={Test_F1:.3f}, AUC={Test_ROC_AUC:.3f}, "
          "threshold={Threshold:.2f}".format(**best_row))
    print("Results:", RESULT_PATH)
    print("Model:", MODEL_PATH)


if __name__ == "__main__":
    main()
