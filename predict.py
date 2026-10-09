"""
predict.py
==========
CLI prediction using the final unified pipeline.
Run improved_pipeline.py first to generate models/final_defect_prediction_pipeline.pkl
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib

BASE          = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE, "models", "final_defect_prediction_pipeline.pkl")
METADATA_PATH = os.path.join(BASE, "models", "final_metadata.json")

CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]
CK_DESC = {
    "wmc":  "WMC  - Weighted Methods per Class",
    "dit":  "DIT  - Depth of Inheritance Tree",
    "noc":  "NOC  - Number of Children",
    "cbo":  "CBO  - Coupling Between Objects",
    "rfc":  "RFC  - Response for a Class",
    "lcom": "LCOM - Lack of Cohesion in Methods",
    "loc":  "LOC  - Lines of Code",
}


def engineer_features(d):
    """Must exactly match improved_pipeline.py engineer_features()."""
    eps = 1e-6
    for col in ["wmc", "loc", "lcom", "rfc", "cbo", "dit"]:
        if col in d:
            d[f"log_{col}"] = float(np.log1p(max(d[col], 0)))
    if "wmc" in d and "loc" in d:
        d["wmc_per_loc"]  = d["wmc"]  / (d["loc"]  + eps)
    if "cbo" in d and "loc" in d:
        d["cbo_per_loc"]  = d["cbo"]  / (d["loc"]  + eps)
    if "rfc" in d and "wmc" in d:
        d["rfc_per_wmc"]  = d["rfc"]  / (d["wmc"]  + eps)
    if "lcom" in d and "wmc" in d:
        d["lcom_per_wmc"] = d["lcom"] / (d["wmc"]  + eps)
    if "cbo" in d and "rfc" in d:
        d["coupling_sum"] = d["cbo"]  + d["rfc"]
    if all(c in d for c in ["wmc", "rfc", "cbo"]):
        d["complexity_avg"] = (d["wmc"] + d["rfc"] + d["cbo"]) / 3.0
    return d


def load_artifacts():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Pipeline not found: {PIPELINE_PATH}\n"
            "Run: python improved_pipeline.py"
        )
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as fh:
            metadata = json.load(fh)
    return pipeline, metadata


def get_user_input():
    print("\nEnter CK metric values:")
    print("-" * 50)
    metrics = {}
    for feat in CK_FEATURES:
        while True:
            try:
                val = float(input(f"  {CK_DESC[feat]} : ").strip())
                if val < 0:
                    raise ValueError("Must be >= 0")
                metrics[feat] = val
                break
            except ValueError as e:
                print(f"    [Invalid] {e}")
    return metrics


def predict(pipeline, metadata, metrics):
    raw = dict(metrics)
    raw = engineer_features(raw)

    feat_cols = metadata.get("training_features", list(raw.keys()))
    missing_features = [feature for feature in feat_cols if feature not in raw]
    if missing_features:
        raise ValueError(
            "The model requires features that cannot be derived from the "
            "seven CK metrics: " + ", ".join(missing_features)
            + ". Retrain the model with improved_pipeline.py or supply all "
              "required metrics."
        )
    row = {f: raw[f] for f in feat_cols}
    X = pd.DataFrame([row], columns=feat_cols).values.astype(float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    threshold = metadata.get("threshold", 0.5)
    prob = float(pipeline.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)

    risk = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.45 else "LOW")
    return pred, prob, risk, threshold


def main():
    print("=" * 50)
    print("   Software Defect Prediction")
    print("=" * 50)

    try:
        pipeline, metadata = load_artifacts()
        print(f"Model loaded: {metadata.get('model', type(pipeline).__name__)}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    try:
        metrics = get_user_input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    pred, prob, risk, thr = predict(pipeline, metadata, metrics)
    label = "DEFECTIVE" if pred == 1 else "CLEAN"

    print("\n" + "=" * 40)
    print("         PREDICTION RESULTS")
    print("=" * 40)
    print(f"Prediction  : {label}")
    print(f"Probability : {prob:.4f}")
    print(f"Risk Level  : {risk}")
    print(f"Threshold   : {thr:.2f}")
    print("=" * 40)

    if pred == 1:
        msgs = {
            "HIGH":   "High risk — immediate code review recommended.",
            "MEDIUM": "Moderate risk — thorough review advised.",
            "LOW":    "Low-moderate risk — consider targeted inspection.",
        }
        print(f">> {msgs[risk]}")
    else:
        print(">> Module appears clean." if prob <= 0.25
              else ">> Borderline clean — light review recommended.")


if __name__ == "__main__":
    main()
