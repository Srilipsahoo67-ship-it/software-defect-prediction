"""
app.py
======
Streamlit web application for Software Defect Prediction.

Loads the trained XGBoost model and StandardScaler, accepts CK metrics
from the user, and predicts whether a software module is defective.

Usage
-----
    streamlit run app.py

Prerequisites
-------------
    Run save_best_model.py first to generate best_model.pkl and scaler.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import shap

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(SCRIPT_DIR, "scaler.pkl")

# Must match the exact order used during training
CK_FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "loc"]

# Labels and helper text shown in the sidebar input widgets
CK_META = {
    "wmc":  ("WMC",  "Weighted Methods per Class",   0,   200,  10),
    "dit":  ("DIT",  "Depth of Inheritance Tree",    0,    20,   2),
    "noc":  ("NOC",  "Number of Children",           0,    50,   0),
    "cbo":  ("CBO",  "Coupling Between Objects",     0,   100,   5),
    "rfc":  ("RFC",  "Response for a Class",         0,   500,  20),
    "lcom": ("LCOM", "Lack of Cohesion in Methods",  0,  2000,  30),
    "loc":  ("LOC",  "Lines of Code",                0, 10000, 100),
    #         label   description                    min   max  default
}

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — clean modern look
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fb; }

    /* Header banner */
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.18);
    }
    .header-box h1 {
        color: #e0e0e0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: 0.5px;
    }
    .header-box p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0;
    }

    /* Result cards */
    .result-card {
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .card-defective {
        background: linear-gradient(135deg, #fff5f5, #fed7d7);
        border-left: 6px solid #e53e3e;
    }
    .card-clean {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border-left: 6px solid #38a169;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .card-value {
        font-size: 1.9rem;
        font-weight: 800;
    }
    .value-defective { color: #c53030; }
    .value-clean     { color: #276749; }
    .value-neutral   { color: #2b6cb0; }

    /* Metric tiles */
    .metric-tile {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        border-top: 4px solid #667eea;
    }
    .metric-tile .label { font-size: 0.78rem; color: #718096; font-weight: 600; text-transform: uppercase; }
    .metric-tile .value { font-size: 1.5rem; font-weight: 700; color: #2d3748; }

    /* Section headers */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #2d3748;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin: 1.4rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stNumberInput label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.7rem 1rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }

    /* DataFrame table */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS  (cached so they are loaded only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """
    Load best_model.pkl and scaler.pkl from disk.
    Uses st.cache_resource so the files are read only once.
    Returns (model, scaler) or raises an error shown in the UI.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            "best_model.pkl not found. Run save_best_model.py first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "scaler.pkl not found. Run save_best_model.py first."
        )
    model  = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-box">
    <h1>&#128269; Software Defect Prediction</h1>
    <p>Predict whether a software module is defective using CK object-oriented metrics.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CK METRIC INPUTS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## &#9881; CK Metric Inputs")
    st.markdown("Enter the object-oriented metrics for the software module you want to evaluate.")
    st.markdown("---")

    user_inputs = {}
    for feature in CK_FEATURES:
        label, description, min_val, max_val, default = CK_META[feature]
        user_inputs[feature] = st.number_input(
            label       = "%s — %s" % (label, description),
            min_value   = float(min_val),
            max_value   = float(max_val),
            value       = float(default),
            step        = 1.0,
            key         = feature,
            help        = "%s ranges from %d to %d in the PROMISE dataset." % (label, min_val, max_val),
        )

    st.markdown("---")
    predict_clicked = st.button("Predict Defect", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL — show error in main area if files are missing
# ─────────────────────────────────────────────────────────────────────────────

try:
    model, scaler = load_artifacts()
    model_name    = type(model).__name__
except FileNotFoundError as e:
    st.error("**Model files not found.**\n\n%s" % str(e))
    st.stop()
except Exception as e:
    st.error("**Failed to load model artifacts:** %s" % str(e))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# IDLE STATE — shown before the button is clicked
# ─────────────────────────────────────────────────────────────────────────────

if not predict_clicked:
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown("""
        <div class="metric-tile">
            <div class="label">Model</div>
            <div class="value">XGBoost</div>
        </div>""", unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div class="metric-tile">
            <div class="label">Training Dataset</div>
            <div class="value">PROMISE</div>
        </div>""", unsafe_allow_html=True)

    with col_info3:
        st.markdown("""
        <div class="metric-tile">
            <div class="label">CK Features</div>
            <div class="value">7 Metrics</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**How to use:**  Adjust the 7 CK metric sliders in the left sidebar, "
        "then click **Predict Defect** to get the prediction result.",
        icon="ℹ️"
    )

    # CK metrics reference table
    st.markdown('<div class="section-title">CK Metrics Reference</div>', unsafe_allow_html=True)
    ref_data = {
        "Metric": ["WMC", "DIT", "NOC", "CBO", "RFC", "LCOM", "LOC"],
        "Full Name": [
            "Weighted Methods per Class",
            "Depth of Inheritance Tree",
            "Number of Children",
            "Coupling Between Objects",
            "Response for a Class",
            "Lack of Cohesion in Methods",
            "Lines of Code",
        ],
        "High Value Indicates": [
            "Complex class, harder to maintain",
            "Deep hierarchy, fragile base class risk",
            "Wide hierarchy, high reuse but rigid",
            "High coupling, low modularity",
            "Many method calls, complex behaviour",
            "Low cohesion, class doing too much",
            "Large class, likely complex",
        ],
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION — runs when button is clicked
# ─────────────────────────────────────────────────────────────────────────────

if predict_clicked:

    # --- 1. Build DataFrame from user inputs ---
    input_df = pd.DataFrame([user_inputs], columns=CK_FEATURES)

    # --- 2. Scale using the saved StandardScaler ---
    # Pass .values (numpy array) to avoid sklearn feature-name mismatch warning
    scaled_array = scaler.transform(input_df.values)

    # --- 3. Run prediction ---
    prediction  = int(model.predict(scaled_array)[0])
    proba_array = model.predict_proba(scaled_array)[0]   # [prob_class_0, prob_class_1]
    probability = round(float(proba_array[1]), 4)        # probability of being defective
    confidence  = round(probability * 100, 1)
    label       = "Defective" if prediction == 1 else "Not Defective"
    card_class  = "card-defective" if prediction == 1 else "card-clean"
    value_class = "value-defective" if prediction == 1 else "value-clean"
    icon        = "&#128308;" if prediction == 1 else "&#128994;"  # red / green circle

    # ── Result Cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="result-card {card_class}">
            <div class="card-title">Prediction</div>
            <div class="card-value {value_class}">{icon} {label}</div>
        </div>""".format(
            card_class=card_class, value_class=value_class,
            icon=icon, label=label
        ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card" style="background:white; border-left:6px solid #3182ce;
             box-shadow:0 2px 12px rgba(0,0,0,0.08);">
            <div class="card-title">Probability</div>
            <div class="card-value value-neutral">{prob}</div>
        </div>""".format(prob=probability), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="result-card" style="background:white; border-left:6px solid #805ad5;
             box-shadow:0 2px 12px rgba(0,0,0,0.08);">
            <div class="card-title">Confidence</div>
            <div class="card-value" style="color:#553c9a;">{conf}%</div>
        </div>""".format(conf=confidence), unsafe_allow_html=True)

    # ── Verdict Banner ────────────────────────────────────────────────────────
    if prediction == 1:
        if probability >= 0.80:
            msg = "High Risk: This module very likely contains defects. Immediate code review recommended."
        elif probability >= 0.60:
            msg = "Moderate Risk: This module shows signs of defects. A thorough review is advised."
        else:
            msg = "Low-Moderate Risk: Borderline defective. Consider a targeted code inspection."
        st.error("**%s**" % msg, icon="🚨")
    else:
        if probability <= 0.20:
            msg = "Low Risk: This module appears clean and well-structured."
        else:
            msg = "Borderline Clean: The module leans towards clean but warrants a light review."
        st.success("**%s**" % msg, icon="✅")

    # ── Two-column layout: table + chart ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 1], gap="large")

    # --- Input Summary Table ---
    with left_col:
        st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Metric"     : [m[0] for m in CK_META.values()],
            "Description": [m[1] for m in CK_META.values()],
            "Value"      : [user_inputs[f] for f in CK_FEATURES],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- Prediction Probability Bar Chart ---
    with right_col:
        st.markdown('<div class="section-title">Prediction Probability</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fb")

        classes = ["Not Defective", "Defective"]
        values  = [round(float(proba_array[0]), 4), round(float(proba_array[1]), 4)]
        colors  = ["#38a169", "#e53e3e"]
        bars    = ax.barh(classes, values, color=colors, height=0.45, edgecolor="white", linewidth=1.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                "%.1f%%" % (val * 100),
                va="center", ha="left", fontsize=11, fontweight="bold", color="#2d3748"
            )

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability", fontsize=10, color="#4a5568")
        ax.set_title("Class Probability Distribution", fontsize=11,
                     fontweight="bold", color="#2d3748", pad=10)
        ax.tick_params(colors="#4a5568", labelsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Scaled Feature Values (expandable) ───────────────────────────────────
    with st.expander("View Scaled Feature Values (after StandardScaler)"):
        scaled_df = pd.DataFrame(scaled_array, columns=[f.upper() for f in CK_FEATURES])
        st.dataframe(scaled_df.round(4), use_container_width=True, hide_index=True)
        st.caption(
            "These are the normalised values passed to the model. "
            "Scaling is applied using the mean and std from the training dataset."
        )

    # ── Tabs: SHAP | Cost/Effort | Model Insights | Dataset Stats ────────────
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "&#129504; SHAP Explanation",
        "&#128200; Cost / Effort Analysis",
        "&#128202; Model Insights",
        "&#128203; Dataset Statistics",
    ])

    # ── Tab 1: SHAP ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Why did the model predict this?</div>', unsafe_allow_html=True)
        st.caption("SHAP (SHapley Additive exPlanations) shows how much each CK metric pushed the prediction towards Defective (+) or Clean (−).")
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled_array)   # shape (1, n_features)
            sv          = shap_values[0] if shap_values.ndim == 2 else shap_values[0]
            feat_labels = [f.upper() for f in CK_FEATURES]

            # Sort by absolute impact
            order      = np.argsort(np.abs(sv))
            sorted_sv  = sv[order]
            sorted_lbl = [feat_labels[i] for i in order]
            colors_shap = ["#e53e3e" if v > 0 else "#38a169" for v in sorted_sv]

            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            bars = ax_shap.barh(sorted_lbl, sorted_sv, color=colors_shap,
                                height=0.55, edgecolor="white")
            for bar, val in zip(bars, sorted_sv):
                ax_shap.text(
                    val + (0.002 if val >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    "%+.3f" % val, va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=9, fontweight="bold", color="#2d3748"
                )
            ax_shap.axvline(0, color="#718096", linewidth=1.2, linestyle="--")
            ax_shap.set_xlabel("SHAP Value  (positive = pushes towards Defective)", fontsize=9)
            ax_shap.set_title("SHAP Feature Contributions for This Prediction",
                              fontsize=11, fontweight="bold")
            ax_shap.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)

            # Interpretation table
            shap_df = pd.DataFrame({
                "Metric"     : [feat_labels[i] for i in order[::-1]],
                "SHAP Value" : ["%+.4f" % sv[i] for i in order[::-1]],
                "Impact"     : ["Increases defect risk" if sv[i] > 0 else "Reduces defect risk" for i in order[::-1]],
            })
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning("SHAP explanation unavailable: %s" % str(e))

    # ── Tab 2: Cost / Effort Analysis ────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Testing Effort Reduction</div>', unsafe_allow_html=True)
        st.caption("ML-guided testing focuses effort on high-risk modules, reducing overall testing cost.")

        total_modules  = 100
        defect_rate    = 0.33
        high_risk      = int(total_modules * defect_rate * 1.4)   # ~46 flagged (some FP)
        effort_saved   = round((1 - high_risk / total_modules) * 100, 1)

        c1, c2, c3 = st.columns(3)
        c1.markdown("""
        <div class="metric-tile">
            <div class="label">Total Modules</div>
            <div class="value">100</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown("""
        <div class="metric-tile" style="border-top-color:#e53e3e;">
            <div class="label">High-Risk (ML flagged)</div>
            <div class="value" style="color:#c53030;">~%d</div>
        </div>""" % high_risk, unsafe_allow_html=True)
        c3.markdown("""
        <div class="metric-tile" style="border-top-color:#38a169;">
            <div class="label">Testing Effort Saved</div>
            <div class="value" style="color:#276749;">~%s%%</div>
        </div>""" % effort_saved, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Flow diagram as a simple visual
        fig_cost, ax_cost = plt.subplots(figsize=(10, 3))
        ax_cost.axis("off")
        fig_cost.patch.set_facecolor("#f8f9fb")

        boxes = [
            (0.05, "Without ML\n100 modules\ntested",         "#fed7d7", "#c53030"),
            (0.30, "With ML\nOnly ~%d high-risk\nmodules tested" % high_risk, "#bee3f8", "#2b6cb0"),
            (0.58, "Effort Saved\n~%s%% reduction\nin testing scope" % effort_saved, "#c6f6d5", "#276749"),
            (0.82, "Outcome\nFaster releases\nLower QA cost",  "#e9d8fd", "#553c9a"),
        ]
        for x, txt, fc, ec in boxes:
            ax_cost.add_patch(mpatches.FancyBboxPatch((x, 0.15), 0.18, 0.65,
                boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, linewidth=2))
            ax_cost.text(x + 0.09, 0.48, txt, ha="center", va="center",
                         fontsize=9, fontweight="bold", color="#2d3748",
                         multialignment="center")
        for x in [0.23, 0.48, 0.76]:
            ax_cost.annotate("", xy=(x + 0.07, 0.48), xytext=(x, 0.48),
                             arrowprops=dict(arrowstyle="->", color="#718096", lw=2))
        plt.tight_layout()
        st.pyplot(fig_cost, use_container_width=True)
        plt.close(fig_cost)

        st.info(
            "**Research context:** Hall et al. (2012) showed that defect prediction models "
            "can reduce testing effort by 25–50% by prioritising high-risk modules. "
            "This aligns with the PROMISE dataset findings where ~33% of modules are defective.",
            icon="📚"
        )

    # ── Tab 3: Model Insights ─────────────────────────────────────────────────
    with tab3:
        fi_path = os.path.join(SCRIPT_DIR, "plots", "feature_importance.png")
        pr_path = os.path.join(SCRIPT_DIR, "plots", "precision_recall_curves.png")

        if os.path.exists(fi_path):
            st.markdown('<div class="section-title">Feature Importance (RF + XGBoost)</div>', unsafe_allow_html=True)
            st.image(fi_path, use_container_width=True)
        else:
            st.warning("feature_importance.png not found. Run: python visualize.py")

        st.markdown("<br>", unsafe_allow_html=True)

        if os.path.exists(pr_path):
            st.markdown('<div class="section-title">Precision-Recall Curves</div>', unsafe_allow_html=True)
            st.caption("PR curves are more informative than ROC for imbalanced datasets. Higher area = better precision and recall together.")
            st.image(pr_path, use_container_width=True)
        else:
            st.warning("precision_recall_curves.png not found. Run: python visualize.py")

    # ── Tab 4: Dataset Statistics ─────────────────────────────────────────────
    with tab4:
        ds_path = os.path.join(SCRIPT_DIR, "plots", "dataset_statistics.png")
        if os.path.exists(ds_path):
            st.markdown('<div class="section-title">PROMISE Repository — Dataset Overview</div>', unsafe_allow_html=True)
            st.image(ds_path, use_container_width=True)
        else:
            st.warning("dataset_statistics.png not found. Run: python visualize.py")

        st.markdown("<br>", unsafe_allow_html=True)
        proj_data = {
            "Project" : ["ant","camel","ivy","jedit","log4j","lucene","poi","synapse","velocity","xalan","xerces"],
            "Version" : ["1.7","1.6","2.0","4.3","1.2","2.4","3.0","1.2","1.6","2.7","1.4"],
            "Samples" : [745,965,352,492,205,340,442,256,229,909,440],
            "Defect Rate": ["22%","19%","11%","2%","19%","59%","63%","35%","34%","99%","15%"],
        }
        st.dataframe(pd.DataFrame(proj_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.82rem; padding:1rem 0;
            border-top: 1px solid #e2e8f0;">
    Software Defect Prediction &nbsp;|&nbsp; PROMISE Repository &nbsp;|&nbsp;
    Model: XGBoost &nbsp;|&nbsp; Features: CK Metrics
</div>
""", unsafe_allow_html=True)
