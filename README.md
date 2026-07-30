# Software Defect Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3-red?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

> **Predict whether a software module is defective using CK object-oriented metrics extracted from Java projects in the PROMISE repository.**

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [CK Metrics Used](#ck-metrics-used)
- [Machine Learning Models](#machine-learning-models)
- [Data Preprocessing](#data-preprocessing)
- [Cross Validation](#cross-validation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Project Overview

Software defect prediction is the process of identifying software modules that are likely to contain bugs **before** testing or deployment. By analysing structural and complexity metrics extracted from source code, machine learning models can learn patterns that distinguish defective modules from clean ones.

This project applies **supervised machine learning** to predict software defects using **CK (Chidamber & Kemerer) object-oriented metrics** collected from real-world Java projects in the **PROMISE repository** — a widely used benchmark in software engineering research.

Early defect prediction enables development teams to:

- **Focus testing efforts** on high-risk modules
- **Reduce costs** by catching bugs earlier in the development cycle
- **Improve software quality** through data-driven code reviews
- **Save time** by prioritising inspection of complex, coupled classes

---

## Objectives

- Build a complete ML pipeline for software defect prediction using CK metrics
- Train and compare multiple classification algorithms on the PROMISE dataset
- Handle class imbalance using **SMOTE** oversampling
- Evaluate models using robust metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Validate results using **5-fold stratified cross-validation**
- Save the best-performing model for production inference
- Build an interactive **Streamlit web application** for real-time prediction with SHAP explainability
- Provide a CLI prediction script for terminal-based usage

---

## Dataset

### Source
The datasets are sourced from the **[PROMISE Repository](http://promise.site.uottawa.ca/SERepository/)** — a public data repository for software engineering research.

- **GitHub Mirror:** [klainfo/DefectData](https://github.com/klainfo/DefectData)
- **Format:** CSV files with CK metrics and a binary defect label
- **Label:** `bug` — `1` = Defective, `0` = Clean

### Java Projects Included

| Project  | Version | Samples | Defect Rate |
|----------|---------|---------|-------------|
| ant      | 1.7     | 745     | ~22%        |
| camel    | 1.6     | 965     | ~19%        |
| ivy      | 2.0     | 352     | ~11%        |
| jedit    | 4.3     | 492     | ~2%         |
| log4j    | 1.2     | 205     | ~19%        |
| lucene   | 2.4     | 340     | ~59%        |
| poi      | 3.0     | 442     | ~63%        |
| synapse  | 1.2     | 256     | ~35%        |
| velocity | 1.6     | 229     | ~34%        |
| xalan    | 2.7     | 909     | ~99%        |
| xerces   | 1.4     | 440     | ~15%        |

**Total: 3,805 samples after deduplication — 2,538 Clean (66.7%), 1,267 Defective (33.3%)**

### Split Strategy

| Split    | Ratio | Samples |
|----------|-------|---------|
| Training | 80%   | 3,044   |
| Testing  | 20%   | 761     |
| Total    | 100%  | 3,805   |

> Stratified splitting is used to preserve the class ratio in both train and test sets.

---

## CK Metrics Used

CK metrics (Chidamber & Kemerer, 1994) are a suite of object-oriented software metrics widely used in defect prediction research.

| Metric | Full Name                    | Description                                                               | High Value Risk              |
|--------|------------------------------|---------------------------------------------------------------------------|------------------------------|
| WMC    | Weighted Methods per Class   | Sum of complexities of all methods in a class.                            | Hard to test and maintain    |
| DIT    | Depth of Inheritance Tree    | Length of the longest path from the class to the root in the hierarchy.   | Fragile base class problem   |
| NOC    | Number of Children           | Number of immediate subclasses of a class.                                | Rigid hierarchy              |
| CBO    | Coupling Between Objects     | Number of classes a class is coupled to.                                  | Low modularity, ripple effects|
| RFC    | Response for a Class         | Number of methods that can be invoked in response to a message.           | Complex behaviour            |
| LCOM   | Lack of Cohesion in Methods  | Measures how unrelated the methods of a class are to each other.          | Class doing too many things  |
| LOC    | Lines of Code                | Total number of lines of source code in the class.                        | Large, complex class         |

---

## Machine Learning Models

| Model          | Type               | Key Strength                               |
|----------------|--------------------|--------------------------------------------|
| Random Forest  | Ensemble (Bagging) | Robust, reduces overfitting                |
| Decision Tree  | Tree-based         | Interpretable rules, handles non-linearity |
| SVM            | Kernel-based       | Effective in high-dimensional spaces       |
| XGBoost        | Ensemble (Boosting)| High performance, handles imbalance well   |

> All models use `random_state=42` for reproducibility.

---

## Data Preprocessing

A clean, leakage-free preprocessing pipeline is applied before training:

1. **Data Loading & Merging** — All 11 CSV files loaded and merged. Column names normalised to lowercase.
2. **Label Normalisation** — `true/yes/1` → 1 (Defective), `false/no/0` → 0 (Clean)
3. **Missing Value Handling** — Column median imputation, robust to outliers in CK metrics.
4. **Duplicate Removal** — Exact duplicate rows dropped to prevent bias.
5. **Feature Scaling** — `StandardScaler` fitted only on the training set, then applied to both sets.
6. **SMOTE** — Applied only to the training set after splitting to avoid data leakage.

```python
scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)   # fit on train only
X_test_sc     = scaler.transform(X_test)        # transform with train stats

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)
```

| Class        | Before SMOTE | After SMOTE |
|--------------|-------------|-------------|
| Clean (0)    | 2,030       | 2,030       |
| Defective (1)| 1,014       | 2,030       |

---

## Cross Validation

**5-Fold Stratified Cross-Validation** is used to produce reliable, unbiased performance estimates.

- **Stratified** — each fold preserves the original class ratio
- **SMOTE inside each fold** — applied only to training folds, never to validation folds
- **Pipeline-based** — `imblearn.pipeline.Pipeline` ensures no leakage

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42)),
    ("model",  classifier),
])
scores = cross_validate(pipe, X, y, cv=skf, scoring=["f1", "roc_auc"])
```

---

## Evaluation Metrics

| Metric           | Why It Matters                                          |
|------------------|---------------------------------------------------------|
| Accuracy         | Overall correctness                                     |
| Precision        | How many predicted defects are real defects             |
| Recall           | How many actual defects were caught                     |
| F1-Score         | Balance between Precision and Recall                    |
| ROC-AUC          | Best metric for imbalanced datasets                     |
| Confusion Matrix | Visualises TP, TN, FP, FN breakdown                     |

> **F1-Score and AUC-ROC are the primary metrics** for this project due to class imbalance.

---

## Results

### 5-Fold Cross-Validation Results (with SMOTE)

| Model         | Accuracy | Precision | Recall | F1-Score | AUC-ROC    |
|---------------|----------|-----------|--------|----------|------------|
| Random Forest | 0.6581   | 0.4837    | 0.4097 | 0.4432   | **0.6525** |
| Decision Tree | 0.6000   | 0.4106    | 0.4625 | 0.4349   | 0.5659     |
| SVM           | 0.6360   | 0.4550    | 0.4507 | 0.4524   | 0.6399     |
| XGBoost       | **0.6602** | **0.4878** | 0.4002 | 0.4392 | 0.6415     |

### Hold-Out Test Set Results (with SMOTE)

| Model         | Accuracy   | Precision  | Recall | F1-Score | AUC-ROC |
|---------------|------------|------------|--------|----------|---------|
| Random Forest | 0.6399     | 0.4507     | 0.3794 | 0.4120   | 0.6248  |
| Decision Tree | 0.5834     | 0.3881     | 0.4387 | 0.4119   | 0.5453  |
| SVM           | 0.6386     | 0.4567     | 0.4585 | **0.4576** | 0.6323 |
| XGBoost       | **0.6570** | **0.4815** | 0.4111 | 0.4435   | **0.6392** |

### Best Model

```
Best Model  : XGBoost
AUC-ROC     : 0.6392  (hold-out) / 0.6415 (5-fold CV)
Accuracy    : 0.6570
Saved as    : best_model.pkl
```

> XGBoost was selected as the production model based on highest AUC-ROC on the hold-out test set.
> SMOTE significantly improved Recall for SVM from 0.05 to 0.46 — a 41-point gain.

---

## Project Structure

```
software-defect-prediction/
│
├── datasets/                    # PROMISE repository CSV files (11 projects)
│   ├── ant-1.7.csv
│   ├── camel-1.6.csv
│   └── ...
│
├── plots/                       # Generated visualisation images
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── dataset_statistics.png
│   └── cv_results.png
│
├── outputs/                     # Generated result CSV files and numpy arrays
│   ├── results_summary.csv
│   ├── results_cv.csv
│   ├── results_smote.csv
│   ├── X_test.npy
│   └── y_test.npy
│
├── best_model.pkl               # Saved best model (XGBoost)
├── scaler.pkl                   # Fitted StandardScaler
│
├── train_models.py              # Train all 4 models with SMOTE, save results
├── validate_pipeline.py         # Full validation: SMOTE leakage check + CV + plots
├── save_best_model.py           # Select and save best model by AUC-ROC
├── visualize.py                 # Generate all 8 visualisation plots
├── predict.py                   # CLI prediction script
├── app.py                       # Streamlit web application with SHAP
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/software-defect-prediction.git
cd software-defect-prediction

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn
scipy
joblib
streamlit
shap
```

---

## How to Run

### 1. Train Models

Trains all 4 models with SMOTE, evaluates on the hold-out test set, saves `outputs/results_summary.csv`.

```bash
python train_models.py
```

### 2. Validate Pipeline

Runs the full validation pipeline — SMOTE leakage check, confusion matrices, ROC curves, feature importance, 5-fold CV.

```bash
python validate_pipeline.py
```

### 3. Save Best Model

Compares all models by AUC-ROC and saves the best one as `best_model.pkl` + `scaler.pkl`.

```bash
python save_best_model.py
```

### 4. Generate Visualisations

Generates all 8 plots and saves them to the `plots/` directory.

```bash
python visualize.py
```

### 5. CLI Prediction

Interactive terminal-based prediction. Enter CK metrics when prompted.

```bash
python predict.py
```

### 6. Run Streamlit Web App

```bash
python -m streamlit run app.py
```

Open your browser at: **http://localhost:8501**

The app includes:
- Real-time defect prediction with probability and confidence
- **SHAP explainability** — shows which CK metrics drove the prediction
- Cost/Effort analysis — testing effort reduction visualisation
- Model insights — feature importance and precision-recall curves
- Dataset statistics — PROMISE repository overview

---

## Future Work

| Area | Description |
|------|-------------|
| Deep Learning | Apply LSTM or Transformer-based models on sequential commit history |
| Hyperparameter Tuning | Extend GridSearchCV / Optuna tuning to all models |
| CI/CD Integration | Embed defect prediction into GitHub Actions pipelines |
| REST API | Wrap the model in a FastAPI or Flask REST endpoint |
| Cloud Deployment | Deploy the Streamlit app on AWS EC2 or Streamlit Cloud |
| More Datasets | Extend to NASA MDP, Eclipse, and Android defect datasets |
| Process Metrics | Add code churn, ownership, and commit history metrics |

---

## References

1. Chidamber, S. R., & Kemerer, C. F. (1994). *A metrics suite for object oriented design*. IEEE Transactions on Software Engineering, 20(6), 476–493.

2. Jureczko, M., & Madeyski, L. (2010). *Towards identifying software project clusters with regard to defect prediction*. PROMISE 2010.

3. Hall, T., Beecham, S., Bowes, D., Gray, D., & Counsell, S. (2012). *A systematic literature review on fault prediction performance in software engineering*. IEEE TSE, 38(6), 1276–1304.

4. PROMISE Repository: http://promise.site.uottawa.ca/SERepository/

5. DefectData GitHub Mirror: https://github.com/klainfo/DefectData

---

## License

```
MIT License

Copyright (c) 2024 Software Defect Prediction Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">
Made with Python, Scikit-Learn, XGBoost, SHAP, and Streamlit
</div>
