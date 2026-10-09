# Major Project Evaluation Guide

## Project title

**Software Defect Prediction Using Machine Learning**

## Abstract

This project investigates whether software metrics can help identify classes likely to contain defects. PROMISE-style datasets provide class-level measurements such as lines of code, coupling, cohesion, inheritance, and complexity, together with a defect label. The implementation supports several classifiers and reports accuracy, precision, recall, F1, and ROC-AUC. The POI 3.0 workflow selects a model and threshold using cross-validation within the training partition, then reports performance on a held-out partition. A separate temporal experiment trains on POI 2.5 and evaluates on POI 3.0 to examine performance across releases.

## Problem and objectives

Defects discovered late in development can increase maintenance effort. A classifier can help prioritize code for review by estimating defect risk from metrics available for each class. The project objectives are to:

1. Load and normalize software metrics and defect labels.
2. Compare established classification model families.
3. Select model settings and a decision threshold using training data only in the POI workflow.
4. Report multiple metrics, including class-sensitive metrics for the imbalanced label.
5. Demonstrate a saved prediction pipeline through a Streamlit interface.

## Data and features

The repository includes multiple project and release CSV files under `datasets/`. The current primary within-release experiment uses `poi-3.0.csv`; the temporal experiment uses `poi-2.5.csv` for training and `poi-3.0.csv` for evaluation. The POI feature set contains up to 20 CK/OO metrics, including WMC, DIT, CBO, RFC, LCOM, LOC, and complexity summaries. The target label is normalized to binary defective/non-defective values by the shared label utility.

Rows represent software classes. Equal feature vectors do not by themselves mean that two rows are duplicate classes, so the POI loader retains distinct rows instead of removing records just because metric values match. Missing or non-numeric feature values are handled in the loader. For exact per-file counts and caveats, inspect the source CSVs and `datasets/README.md`.

## Methodology

### POI 3.0 within-release experiment

`train_poi_model.py` creates a stratified training/holdout partition. Candidate classifiers and feature views are compared using stratified five-fold out-of-fold predictions on the training partition. F1 is the selection objective; the decision threshold is selected from those training predictions. The chosen candidate is fitted on the training partition and evaluated on the holdout. The summary is saved to `outputs/poi_f1_results_seed2026.csv`; the selected model bundle is saved to `models/poi_f1_model_seed2026.pkl`.

### Temporal release experiment

`train_poi_temporal.py` uses POI 2.5 for fitting and training-only cross-validation, then evaluates on POI 3.0. This asks a different question from the within-release split. Because POI 3.0 has also been used during prior development/model comparisons, present this result as exploratory and do not describe it as untouched final validation.

### Demo pipeline

`export_poi_xgboost_app.py` exports the XGBoost depth-2 model over all 20 raw POI metrics to `models/poi_xgboost_app.pkl`. It fits only the recorded training partition, carries over the training-CV threshold, and verifies the saved holdout metrics against the existing results CSV. The Streamlit app loads this bundle. `improved_pipeline.py` and `predict.py` remain a separate earlier multi-project workflow.

## Metrics

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** fraction of predicted defective classes that are actually defective.
- **Recall:** fraction of actual defective classes found by the model.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ability to rank positive examples above negative examples across probability cutoffs.
- **Threshold:** probability cutoff used to convert scores into class predictions. It is a setting, not a performance metric.

For defect screening, recall and F1 are useful because missing a defective class can be costly. Precision describes the review burden. Accuracy alone can hide weak performance on the less common class. ROC-AUC is threshold-independent; changing the threshold does not change ROC-AUC.

## Current recorded results

### POI 3.0 within-release holdout

The selected bundle was `Random Forest` using the `CK7` feature view with threshold `0.54`. Its held-out metrics were:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.7753 | 0.8491 | 0.7895 | 0.8182 | 0.8786 |

The app uses the XGBoost depth-2, all-raw-20 candidate. Its held-out accuracy is `0.8427`, precision `0.9216`, recall `0.8246`, F1 `0.8704`, and ROC-AUC `0.9027`. It is not the overall model bundle selected by CV F1; it is the XGBoost candidate with the strongest training-CV accuracy among the XGBoost choices. The held-out result is specific to this recorded split.

### POI 2.5 to POI 3.0 temporal evaluation

The recorded temporal result for ExtraTrees with the raw 20-feature view and threshold `0.47` was accuracy `0.6652`, precision `0.7401`, recall `0.7295`, F1 `0.7348`, and ROC-AUC `0.6683`. Treat this as exploratory given prior development use of POI 3.0.

## Limitations and responsible interpretation

- The data covers a limited set of projects/releases and may not represent other codebases.
- Defect labels and metrics can be noisy or project-specific.
- A held-out score is specific to its split; repeated model comparison against the same holdout can bias the apparent best result.
- Temporal transfer can be difficult because code distributions and defect practices change between releases.
- The threshold should be chosen for the intended review trade-off, not adjusted after looking at the final evaluation labels.
- Do not claim a target score unless the saved result file and evaluation protocol support it.

## Suggested demonstration sequence

1. Introduce the defect prediction problem and define the positive class.
2. Show the dataset files and explain which release pair is used for each experiment.
3. Explain the feature metrics and the preprocessing/label normalization.
4. Walk through training-only cross-validation, threshold selection, and final evaluation.
5. Present the metric table and describe precision/recall trade-offs.
6. Explain why temporal and within-release metrics are not directly interchangeable.
7. Launch the Streamlit demo, enter all 20 metrics, and identify the XGBoost artifact it loads.
8. State limitations and propose future validation on additional projects/releases.

## Run commands

```powershell
python -m pip install -r requirements-app.txt
python export_poi_xgboost_app.py
python -m streamlit run app.py
```

The export command trains the selected candidate on the saved training partition and checks its held-out metrics against the recorded row. It leaves the evaluation CSV unchanged.

