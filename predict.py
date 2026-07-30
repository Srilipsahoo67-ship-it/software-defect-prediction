"""
predict.py
==========
Interactive Software Defect Prediction using saved model artifacts.

Accepts 7 CK metrics from the user via the terminal, scales them using
the fitted StandardScaler, and predicts whether the software module is
Defective or Not Defective along with probability and confidence.

Usage
-----
    python predict.py

Prerequisites
-------------
    Run save_best_model.py first to generate:
        best_model.pkl
        scaler.pkl
"""

import os
import sys
import joblib
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(SCRIPT_DIR, "scaler.pkl")

# Feature order must exactly match what was used during training
CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# Human-readable descriptions shown to the user during input
CK_DESCRIPTIONS = {
    "wmc" : "WMC  - Weighted Methods per Class",
    "dit" : "DIT  - Depth of Inheritance Tree",
    "noc" : "NOC  - Number of Children",
    "cbo" : "CBO  - Coupling Between Objects",
    "rfc" : "RFC  - Response for a Class",
    "lcom": "LCOM - Lack of Cohesion in Methods",
    "loc" : "LOC  - Lines of Code",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    """
    Load best_model.pkl and scaler.pkl from disk.
    Raises FileNotFoundError if either file is missing.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            "best_model.pkl not found at: %s\n"
            "Please run save_best_model.py first." % BEST_MODEL_PATH
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "scaler.pkl not found at: %s\n"
            "Please run save_best_model.py first." % SCALER_PATH
        )

    model  = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Model loaded  : %s" % type(model).__name__)
    print("Scaler loaded : %s" % type(scaler).__name__)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. COLLECT USER INPUT
# ─────────────────────────────────────────────────────────────────────────────

def get_user_input():
    """
    Prompt the user to enter each CK metric value one by one.
    Validates that each input is a non-negative number.
    Returns a dict of {feature_name: float_value}.
    """
    print("\nEnter the CK metric values for the software module:")
    print("-" * 50)

    metrics = {}
    for feature in CK_FEATURES:
        description = CK_DESCRIPTIONS[feature]
        while True:
            try:
                raw = input("  %s : " % description).strip()
                if raw == "":
                    raise ValueError("Input cannot be empty.")
                value = float(raw)
                if value < 0:
                    raise ValueError("Value must be >= 0.")
                metrics[feature] = value
                break
            except ValueError as e:
                print("    [Invalid] %s Please enter a valid non-negative number." % str(e))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESS INPUT
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(metrics, scaler):
    """
    Convert the input dict into a DataFrame and apply the saved StandardScaler.

    Using a DataFrame (instead of a plain array) preserves feature names,
    making the pipeline easier to debug and extend.

    Parameters
    ----------
    metrics : dict  {feature: value}
    scaler  : fitted StandardScaler loaded from scaler.pkl

    Returns
    -------
    scaled DataFrame ready for model inference
    """
    # Build a single-row DataFrame with columns in training order
    df = pd.DataFrame([metrics], columns=CK_FEATURES)

    print("\nRaw input as DataFrame:")
    print(df.to_string(index=False))

    # Scale using training statistics - must use transform(), never fit_transform()
    # Pass .values (numpy array) to avoid sklearn feature-name mismatch warning
    scaled_array = scaler.transform(df.values)
    scaled_df    = pd.DataFrame(scaled_array, columns=CK_FEATURES)

    return scaled_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, scaled_df):
    """
    Run inference on the scaled input and return prediction details.

    Parameters
    ----------
    model     : trained classifier loaded from best_model.pkl
    scaled_df : scaled single-row DataFrame from preprocess()

    Returns
    -------
    dict with keys:
        label       - "Defective" or "Not Defective"
        prediction  - raw int label (1 or 0)
        probability - float probability of being defective (class 1)
        confidence  - probability formatted as a percentage string
    """
    # predict() returns the class label (0 or 1)
    prediction = int(model.predict(scaled_df)[0])

    # predict_proba() returns [prob_class_0, prob_class_1]
    # index [1] gives the probability of being defective
    probability = round(float(model.predict_proba(scaled_df)[0][1]), 4)

    return {
        "label"      : "Defective" if prediction == 1 else "Not Defective",
        "prediction" : prediction,
        "probability": probability,
        "confidence" : "%d%%" % round(probability * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def display_results(result):
    """Print the prediction results in a clean, readable format."""
    print("\n" + "=" * 40)
    print("         PREDICTION RESULTS")
    print("=" * 40)
    print("Prediction  : %s" % result["label"])
    print("Probability : %.2f" % result["probability"])
    print("Confidence  : %s" % result["confidence"])
    print("=" * 40)

    # Extra context based on confidence level
    prob = result["probability"]
    if result["prediction"] == 1:
        if prob >= 0.80:
            print(">> High risk - this module very likely contains defects.")
        elif prob >= 0.60:
            print(">> Moderate risk - review this module carefully.")
        else:
            print(">> Low-moderate risk - consider a code review.")
    else:
        if prob <= 0.20:
            print(">> Low risk - this module appears clean.")
        else:
            print(">> Borderline - the module leans clean but warrants attention.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   Software Defect Prediction")
    print("=" * 50)

    # Step 1 — Load saved artifacts
    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as e:
        print("\n[ERROR] %s" % str(e))
        sys.exit(1)

    # Step 2 — Collect CK metrics from the user
    try:
        metrics = get_user_input()
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled by user.")
        sys.exit(0)

    # Step 3 — Convert to DataFrame and scale
    try:
        scaled_df = preprocess(metrics, scaler)
    except Exception as e:
        print("\n[ERROR] Failed during preprocessing: %s" % str(e))
        sys.exit(1)

    # Step 4 — Run prediction
    try:
        result = predict(model, scaled_df)
    except Exception as e:
        print("\n[ERROR] Failed during prediction: %s" % str(e))
        sys.exit(1)

    # Step 5 — Display results
    display_results(result)


if __name__ == "__main__":
    main()
